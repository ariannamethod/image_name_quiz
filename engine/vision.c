/* vision.c — SmolVLM / idefics3 image preprocessing + SigLIP vision tower. See vision.h. */
#include "vision.h"
#include "notorch_vision.h"   /* nt_image*, stb_image impl lives in THIS TU only */
#include "gguf.h"             /* mmproj weight loading */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

#define TILE     512   /* per-tile / global square size (= vision image_size) */
#define LONGEST 2048   /* outer cap on longest edge before tiling */

static void img_free(nt_image* im) { if (im) { free(im->data); free(im); } }

/* Copy [r0:r0+TILE, c0:c0+TILE] sub-rect of a CHW [0,1] image into dst[3*TILE*TILE].
 * Out-of-bounds (image not a perfect multiple) is zero-padded. */
static void crop_tile(const nt_image* src, int r0, int c0, float* dst) {
    int W = src->width, H = src->height;
    for (int ch = 0; ch < 3; ch++)
        for (int r = 0; r < TILE; r++)
            for (int c = 0; c < TILE; c++) {
                int sr = r0 + r, sc = c0 + c;
                float v = (sr < H && sc < W)
                        ? src->data[(long)ch * H * W + (long)sr * W + sc] : 0.0f;
                dst[(long)ch * TILE * TILE + (long)r * TILE + c] = v;
            }
}

/* In-place normalize a [3,TILE,TILE] frame: (x-0.5)/0.5  (x already in [0,1]). */
static void norm_frame(float* f) {
    long n = 3L * TILE * TILE;
    for (long i = 0; i < n; i++) f[i] = (f[i] - 0.5f) / 0.5f;
}

float* smolvlm_preprocess(const char* path, int* out_n_frames, int* out_S) {
    nt_image* img = nt_image_load(path, 3);          /* CHW float in [0,1] */
    if (!img) return NULL;

    /* cap longest edge to 2048 (preserve aspect) */
    int W = img->width, H = img->height;
    if (W > LONGEST || H > LONGEST) {
        float s = (float)LONGEST / (W > H ? W : H);
        int nw = (int)(W * s + 0.5f), nh = (int)(H * s + 0.5f);
        nt_image* r = nt_image_resize(img, nw, nh);
        img_free(img);
        if (!r) return NULL;
        img = r; W = img->width; H = img->height;
    }

    int n_cols = (W + TILE - 1) / TILE;
    int n_rows = (H + TILE - 1) / TILE;
    int n_tiles = n_rows * n_cols;
    /* Default: NO splitting — matches the oracle (llama-mtmd-cli encodes one
     * global 512x512 frame = 64 tokens; it does not tile SmolVLM by default,
     * despite HF's do_image_splitting=true). Opt-in tiling via SMOLVLM_SPLIT=1
     * (HF-transformers behavior, for large images / higher fidelity). */
    int want_split = (getenv("SMOLVLM_SPLIT") != NULL);
    int splitting = want_split && (n_tiles > 1);
    int n_frames = splitting ? n_tiles + 1 : 1;      /* tiles + one global, or just global */

    float* out = (float*)malloc((long)n_frames * 3 * TILE * TILE * sizeof(float));
    if (!out) { img_free(img); return NULL; }

    int fi = 0;
    if (splitting) {
        /* resize so the image divides evenly into n_rows x n_cols tiles of TILE */
        nt_image* grid = nt_image_resize(img, n_cols * TILE, n_rows * TILE);
        if (!grid) { free(out); img_free(img); return NULL; }
        for (int rr = 0; rr < n_rows; rr++)
            for (int cc = 0; cc < n_cols; cc++)
                crop_tile(grid, rr * TILE, cc * TILE, out + (long)(fi++) * 3 * TILE * TILE);
        img_free(grid);
    }

    /* global image: full (capped) frame resized to TILE x TILE, appended last */
    nt_image* g = nt_image_resize(img, TILE, TILE);
    if (!g) { free(out); img_free(img); return NULL; }
    memcpy(out + (long)fi * 3 * TILE * TILE, g->data, (long)3 * TILE * TILE * sizeof(float));
    fi++;
    img_free(g);
    img_free(img);

    for (int i = 0; i < n_frames; i++) norm_frame(out + (long)i * 3 * TILE * TILE);

    *out_n_frames = n_frames;
    *out_S = TILE;
    return out;
}

/* ─────────────────────────────────────────────────────────────────────────
 * SigLIP vision tower (PHASE 3). Forward verified vs llama.cpp clip.cpp + HF.
 * Raw-pointer f32 (notorch tape ops don't fit a forward loop); BLAS for matmuls.
 * ───────────────────────────────────────────────────────────────────────── */

/* y[m,n] = x[m,k] @ W[n,k]^T   (W is GGUF [out=n, in=k] row-major; PyTorch Linear) */
static void mmT(float* C, const float* A, const float* W, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1.0f, A, k, W, k, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
        float s = 0; for (int p = 0; p < k; p++) s += A[(long)i*k+p] * W[(long)j*k+p]; C[(long)i*n+j] = s; }
#endif
}
/* C[m,n] = A[m,k] @ B[k,n] */
static void mm(float* C, const float* A, const float* B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, A, k, B, n, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) {
        float s = 0; for (int p = 0; p < k; p++) s += A[(long)i*k+p] * B[(long)p*n+j]; C[(long)i*n+j] = s; }
#endif
}
static void add_bias_rows(float* x, const float* b, int m, int n) {
    if (!b) return;
    for (int i = 0; i < m; i++) { float* r = x + (long)i*n; for (int j = 0; j < n; j++) r[j] += b[j]; }
}
/* LayerNorm (mean/var) per row, eps; ggml NORM_TYPE_NORMAL == HF nn.LayerNorm */
static void layernorm_rows(float* x, const float* g, const float* b, int m, int n, float eps) {
    for (int i = 0; i < m; i++) {
        float* r = x + (long)i*n;
        double mu = 0; for (int j = 0; j < n; j++) mu += r[j]; mu /= n;
        double var = 0; for (int j = 0; j < n; j++) { double d = r[j]-mu; var += d*d; } var /= n;
        float inv = 1.0f / sqrtf((float)var + eps);
        for (int j = 0; j < n; j++) r[j] = ((r[j]-(float)mu)*inv)*g[j] + b[j];
    }
}
/* gelu_pytorch_tanh (exact tanh formula; ggml CPU uses an f16 LUT -> ~1e-3 rel drift) */
static void gelu_tanh_inplace(float* x, long n) {
    const float c = 0.79788456080286535588f, a = 0.044715f;
    for (long i = 0; i < n; i++) { float v = x[i]; x[i] = 0.5f*v*(1.0f + tanhf(c*(v + a*v*v*v))); }
}
static void softmax_row(float* x, int n) {
    float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0; for (int i = 0; i < n; i++) { x[i] = expf(x[i]-mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

struct siglip_model {
    int n_layers, hidden, n_heads, head_dim, ffn, patches, patch, img;
    float eps;
    float *patch_w, *patch_b;      /* [hidden, 3*patch*patch], [hidden] */
    float *pos_w;                  /* [patches, hidden] */
    float *post_ln_w, *post_ln_b;  /* [hidden] */
    struct {
        float *ln1_w,*ln1_b,*ln2_w,*ln2_b;
        float *q_w,*q_b,*k_w,*k_b,*v_w,*v_b,*o_w,*o_b;
        float *up_w,*up_b,*dn_w,*dn_b;
    } *L;
};

static float* gd(gguf_file* gf, const char* name) {
    int ti = gguf_find_tensor(gf, name);
    if (ti < 0) return NULL;
    return gguf_dequant(gf, ti);
}
static int kv_u32(gguf_file* gf, const char* key, int def) {
    const gguf_kv* kv = gguf_get_kv(gf, key);
    return kv ? (int)kv->val.u32 : def;
}
static float kv_f32(gguf_file* gf, const char* key, float def) {
    const gguf_kv* kv = gguf_get_kv(gf, key);
    return kv ? kv->val.f32 : def;
}

siglip_model* siglip_load(const char* mmproj_path) {
    gguf_file* gf = gguf_open(mmproj_path);
    if (!gf) return NULL;
    siglip_model* m = (siglip_model*)calloc(1, sizeof(*m));
    if (!m) { gguf_close(gf); return NULL; }

    m->hidden   = kv_u32(gf, "clip.vision.embedding_length", 768);
    m->n_layers = kv_u32(gf, "clip.vision.block_count", 12);
    m->n_heads  = kv_u32(gf, "clip.vision.attention.head_count", 12);
    m->ffn      = kv_u32(gf, "clip.vision.feed_forward_length", 3072);
    m->img      = kv_u32(gf, "clip.vision.image_size", 512);
    m->patch    = kv_u32(gf, "clip.vision.patch_size", 16);
    m->eps      = kv_f32(gf, "clip.vision.attention.layer_norm_epsilon", 1e-6f);
    m->head_dim = m->hidden / m->n_heads;
    int grid    = m->img / m->patch;
    m->patches  = grid * grid;

    printf("siglip: L=%d D=%d heads=%d hd=%d ffn=%d img=%d patch=%d patches=%d eps=%.0e\n",
           m->n_layers, m->hidden, m->n_heads, m->head_dim, m->ffn, m->img, m->patch, m->patches, m->eps);

    m->patch_w = gd(gf, "v.patch_embd.weight");
    m->patch_b = gd(gf, "v.patch_embd.bias");
    m->pos_w   = gd(gf, "v.position_embd.weight");
    m->post_ln_w = gd(gf, "v.post_ln.weight");
    m->post_ln_b = gd(gf, "v.post_ln.bias");

    m->L = (typeof(m->L))calloc(m->n_layers, sizeof(*m->L));
    char nm[128];
    for (int l = 0; l < m->n_layers; l++) {
        #define LD(field, fmt) do { snprintf(nm,sizeof(nm),fmt,l); m->L[l].field = gd(gf, nm); } while(0)
        LD(ln1_w,"v.blk.%d.ln1.weight"); LD(ln1_b,"v.blk.%d.ln1.bias");
        LD(ln2_w,"v.blk.%d.ln2.weight"); LD(ln2_b,"v.blk.%d.ln2.bias");
        LD(q_w,"v.blk.%d.attn_q.weight"); LD(q_b,"v.blk.%d.attn_q.bias");
        LD(k_w,"v.blk.%d.attn_k.weight"); LD(k_b,"v.blk.%d.attn_k.bias");
        LD(v_w,"v.blk.%d.attn_v.weight"); LD(v_b,"v.blk.%d.attn_v.bias");
        LD(o_w,"v.blk.%d.attn_out.weight"); LD(o_b,"v.blk.%d.attn_out.bias");
        LD(up_w,"v.blk.%d.ffn_up.weight"); LD(up_b,"v.blk.%d.ffn_up.bias");
        LD(dn_w,"v.blk.%d.ffn_down.weight"); LD(dn_b,"v.blk.%d.ffn_down.bias");
        #undef LD
    }
    gguf_close(gf);   /* dequant copied to float; raw gguf no longer needed */

    if (!m->patch_w || !m->pos_w || !m->post_ln_w || !m->L[0].q_w) {
        fprintf(stderr, "siglip: missing critical vision weights\n");
        siglip_free(m); return NULL;
    }
    return m;
}

void siglip_free(siglip_model* m) {
    if (!m) return;
    free(m->patch_w); free(m->patch_b); free(m->pos_w); free(m->post_ln_w); free(m->post_ln_b);
    if (m->L) for (int l = 0; l < m->n_layers; l++) {
        free(m->L[l].ln1_w); free(m->L[l].ln1_b); free(m->L[l].ln2_w); free(m->L[l].ln2_b);
        free(m->L[l].q_w); free(m->L[l].q_b); free(m->L[l].k_w); free(m->L[l].k_b);
        free(m->L[l].v_w); free(m->L[l].v_b); free(m->L[l].o_w); free(m->L[l].o_b);
        free(m->L[l].up_w); free(m->L[l].up_b); free(m->L[l].dn_w); free(m->L[l].dn_b);
    }
    free(m->L); free(m);
}

int siglip_n_patches(const siglip_model* m) { return m ? m->patches : 0; }
int siglip_hidden(const siglip_model* m)    { return m ? m->hidden  : 0; }

int siglip_encode(const siglip_model* m, const float* frame, float* out) {
    int P = m->patches, D = m->hidden, H = m->n_heads, hd = m->head_dim;
    int grid = m->img / m->patch, ps = m->patch, img = m->img, in = 3*ps*ps;
    float scale = 1.0f / sqrtf((float)hd);

    /* 1) patch embed: build X[P, in] in (c,kh,kw) order, then X @ patch_w^T + bias */
    float* X = (float*)malloc((long)P*in*sizeof(float));
    if (!X) return -1;
    for (int ph = 0; ph < grid; ph++) for (int pw = 0; pw < grid; pw++) {
        int p = ph*grid + pw;             /* row-major patch index (h outer, w inner) */
        float* xr = X + (long)p*in;
        int t = 0;
        for (int c = 0; c < 3; c++) for (int kh = 0; kh < ps; kh++) for (int kw = 0; kw < ps; kw++)
            xr[t++] = frame[(long)c*img*img + (long)(ph*ps+kh)*img + (pw*ps+kw)];
    }
    mmT(out, X, m->patch_w, P, in, D);     /* out = hidden [P, D] */
    free(X);
    add_bias_rows(out, m->patch_b, P, D);

    /* 2) + position embedding */
    for (long i = 0; i < (long)P*D; i++) out[i] += m->pos_w[i];

    /* scratch */
    float* tmp = (float*)malloc((long)P*D*sizeof(float));
    float* q   = (float*)malloc((long)P*D*sizeof(float));
    float* k   = (float*)malloc((long)P*D*sizeof(float));
    float* v   = (float*)malloc((long)P*D*sizeof(float));
    float* att = (float*)malloc((long)P*D*sizeof(float));
    float* proj= (float*)malloc((long)P*D*sizeof(float));
    float* up  = (float*)malloc((long)P*m->ffn*sizeof(float));
    float* qh  = (float*)malloc((long)P*hd*sizeof(float));
    float* kh_ = (float*)malloc((long)P*hd*sizeof(float));
    float* vh  = (float*)malloc((long)P*hd*sizeof(float));
    float* sc  = (float*)malloc((long)P*P*sizeof(float));
    float* oh  = (float*)malloc((long)P*hd*sizeof(float));
    if (!tmp||!q||!k||!v||!att||!proj||!up||!qh||!kh_||!vh||!sc||!oh) {
        free(tmp);free(q);free(k);free(v);free(att);free(proj);free(up);free(qh);free(kh_);free(vh);free(sc);free(oh);
        return -1;
    }

    for (int l = 0; l < m->n_layers; l++) {
        /* ── attention sublayer: x = x + Attn(LN1(x)) ── */
        memcpy(tmp, out, (long)P*D*sizeof(float));
        layernorm_rows(tmp, m->L[l].ln1_w, m->L[l].ln1_b, P, D, m->eps);
        mmT(q, tmp, m->L[l].q_w, P, D, D); add_bias_rows(q, m->L[l].q_b, P, D);
        mmT(k, tmp, m->L[l].k_w, P, D, D); add_bias_rows(k, m->L[l].k_b, P, D);
        mmT(v, tmp, m->L[l].v_w, P, D, D); add_bias_rows(v, m->L[l].v_b, P, D);
        for (int h = 0; h < H; h++) {
            int off = h*hd;
            for (int t = 0; t < P; t++) {
                memcpy(qh +(long)t*hd, q +(long)t*D+off, hd*sizeof(float));
                memcpy(kh_+(long)t*hd, k +(long)t*D+off, hd*sizeof(float));
                memcpy(vh +(long)t*hd, v +(long)t*D+off, hd*sizeof(float));
            }
            mmT(sc, qh, kh_, P, hd, P);                 /* scores[P,P] = qh @ kh^T */
            for (long i = 0; i < (long)P*P; i++) sc[i] *= scale;
            for (int t = 0; t < P; t++) softmax_row(sc + (long)t*P, P);   /* non-causal */
            mm(oh, sc, vh, P, P, hd);                   /* oh[P,hd] = scores @ vh */
            for (int t = 0; t < P; t++) memcpy(att +(long)t*D+off, oh +(long)t*hd, hd*sizeof(float));
        }
        mmT(proj, att, m->L[l].o_w, P, D, D); add_bias_rows(proj, m->L[l].o_b, P, D);
        for (long i = 0; i < (long)P*D; i++) out[i] += proj[i];

        /* ── mlp sublayer: x = x + MLP(LN2(x)) ──
         * NOTE: this GGUF inverts the SigLIP MLP names. Ground truth = bias sizes
         * in the file: ffn_up.bias=768, ffn_down.bias=3072. So ffn_down is fc1
         * (768->3072, out=3072) and ffn_up is fc2 (3072->768, out=768). */
        memcpy(tmp, out, (long)P*D*sizeof(float));
        layernorm_rows(tmp, m->L[l].ln2_w, m->L[l].ln2_b, P, D, m->eps);
        mmT(up, tmp, m->L[l].dn_w, P, D, m->ffn); add_bias_rows(up, m->L[l].dn_b, P, m->ffn);   /* fc1: 768->3072 */
        gelu_tanh_inplace(up, (long)P*m->ffn);
        mmT(proj, up, m->L[l].up_w, P, m->ffn, D); add_bias_rows(proj, m->L[l].up_b, P, D);      /* fc2: 3072->768 */
        for (long i = 0; i < (long)P*D; i++) out[i] += proj[i];
    }

    /* 3) final post-layernorm over all tokens */
    layernorm_rows(out, m->post_ln_w, m->post_ln_b, P, D, m->eps);

    free(tmp);free(q);free(k);free(v);free(att);free(proj);free(up);free(qh);free(kh_);free(vh);free(sc);free(oh);
    return 0;
}
