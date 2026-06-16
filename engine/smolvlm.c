/*
 * smolvlm.c — SmolVLM-256M inference on notorch, pure C (PHASE 1: text decoder)
 *
 * Goal (full): image + prompt -> text, no cloud/API. Built on vendored notorch.
 * THIS phase: text-only llama decoder on SmolVLM's main GGUF + raw-token-id mode
 *             for isolated verification against llama.cpp (tokenizer comes later).
 *
 * Vision tower (SigLIP), pixel-shuffle connector, vision-token splice: later phases.
 *
 * Build: clang -O2 -DUSE_BLAS -DACCELERATE -o smolvlm smolvlm.c notorch.c gguf.c -lm -framework Accelerate
 * Run:   ./smolvlm <model.gguf> [--ids "1 2 3 ..."] [--prompt "text"] [-n N]
 *
 * Decoding is greedy (argmax) — deterministic, to match llama.cpp --temp 0.
 */

#include "gguf.h"
#include "notorch.h"
#include "bpe.h"
#include "vision.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

/* ── math (mirrors examples/infer_llama.c, the verified llama decode path) ──── */

// C[m,n] = A[m,k] @ B^T[n,k]   (B is GGUF weight layout: [out, in] row-major)
static void mm_t(float *C, const float *A, const float *B, int m, int k, int n) {
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                m, n, k, 1.0f, A, k, B, k, 0.0f, C, n);
#else
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0;
            for (int p = 0; p < k; p++) s += A[i*k+p] * B[j*k+p];
            C[i*n+j] = s;
        }
#endif
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * inv;
}

static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float s = 0;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); s += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= s;
}

// RoPE interleaved (2i,2i+1) — correct for GGUF-converted llama (weights permuted at convert)
static void rope(float *x, int pos, int head_dim, float freq_base) {
    for (int i = 0; i < head_dim / 2; i++) {
        float freq = 1.0f / powf(freq_base, 2.0f * i / head_dim);
        float angle = pos * freq;
        float cs = cosf(angle), sn = sinf(angle);
        float x0 = x[2*i], x1 = x[2*i+1];
        x[2*i]   = x0 * cs - x1 * sn;
        x[2*i+1] = x0 * sn + x1 * cs;
    }
}

static void add_bias(float *x, const float *bias, int n) {
    if (bias) for (int i = 0; i < n; i++) x[i] += bias[i];
}

/* ── model ──────────────────────────────────────────────────────────────────── */

typedef struct {
    int n_layers, n_heads, n_kv_heads, embed, ffn, vocab, head_dim, kv_dim, q_dim;
    float rope_base, rms_eps;
    int has_output_weight;

    float *tok_emb, *out_norm, *out_weight;
    struct {
        float *attn_norm, *wq, *wk, *wv, *wo;
        float *q_bias, *k_bias, *v_bias;
        float *ffn_norm, *wgate, *wup, *wdown;
    } layers[];
} llama_model;

static llama_model* llama_load(gguf_file* gf) {
    int nl = gf->n_layers;
    llama_model* m = (llama_model*)calloc(1, sizeof(llama_model) + nl * sizeof(m->layers[0]));
    if (!m) return NULL;
    m->n_layers = nl;
    m->n_heads = gf->n_heads;
    m->n_kv_heads = gf->n_kv_heads;
    m->embed = gf->embed_dim;
    m->ffn = gf->ffn_dim;
    m->rope_base = gf->rope_freq_base;
    m->rms_eps = gf->rms_eps;

    int ti = gguf_find_tensor(gf, "blk.0.attn_q.weight");
    if (ti >= 0) { m->q_dim = (int)gf->tensors[ti].shape[1]; m->head_dim = m->q_dim / m->n_heads; }
    else { m->head_dim = m->embed / m->n_heads; m->q_dim = m->n_heads * m->head_dim; }
    m->kv_dim = m->head_dim * m->n_kv_heads;

    ti = gguf_find_tensor(gf, "token_embd.weight");
    if (ti >= 0) m->vocab = (int)gf->tensors[ti].shape[1];
    else if (gf->vocab_size > 0) m->vocab = gf->vocab_size;
    else m->vocab = 32000;

    printf("smolvlm: E=%d H=%d KV=%d FFN=%d V=%d L=%d HD=%d Q=%d rope=%.0f rms=%.0e\n",
           m->embed, m->n_heads, m->n_kv_heads, m->ffn, m->vocab, nl, m->head_dim, m->q_dim,
           m->rope_base, m->rms_eps);

    ti = gguf_find_tensor(gf, "token_embd.weight");  if (ti >= 0) m->tok_emb = gguf_dequant(gf, ti);
    ti = gguf_find_tensor(gf, "output_norm.weight");  if (ti >= 0) m->out_norm = gguf_dequant(gf, ti);
    ti = gguf_find_tensor(gf, "output.weight");
    if (ti >= 0) { m->out_weight = gguf_dequant(gf, ti); m->has_output_weight = 1; }

    for (int l = 0; l < nl; l++) {
        char name[128];
        #define L(field, fmt) do { snprintf(name, sizeof(name), fmt, l); \
            ti = gguf_find_tensor(gf, name); if (ti >= 0) m->layers[l].field = gguf_dequant(gf, ti); } while(0)
        L(attn_norm, "blk.%d.attn_norm.weight");
        L(wq, "blk.%d.attn_q.weight"); L(wk, "blk.%d.attn_k.weight");
        L(wv, "blk.%d.attn_v.weight"); L(wo, "blk.%d.attn_output.weight");
        L(q_bias, "blk.%d.attn_q.bias"); L(k_bias, "blk.%d.attn_k.bias"); L(v_bias, "blk.%d.attn_v.bias");
        L(ffn_norm, "blk.%d.ffn_norm.weight");
        L(wgate, "blk.%d.ffn_gate.weight"); L(wup, "blk.%d.ffn_up.weight"); L(wdown, "blk.%d.ffn_down.weight");
        #undef L
    }
    if (!m->tok_emb || !m->out_norm) { fprintf(stderr, "smolvlm: missing critical weights\n"); return NULL; }
    if (!m->has_output_weight) printf("  (tied embeddings)\n");
    return m;
}

/* ── kv cache + forward (single token) ──────────────────────────────────────── */

typedef struct { float *k, *v; int max_seq, n_layers, kv_dim; } kv_cache;

static kv_cache* kv_new(int nl, int max_seq, int kv_dim) {
    kv_cache* kv = (kv_cache*)calloc(1, sizeof(kv_cache));
    kv->k = (float*)calloc((long)nl * max_seq * kv_dim, sizeof(float));
    kv->v = (float*)calloc((long)nl * max_seq * kv_dim, sizeof(float));
    kv->max_seq = max_seq; kv->n_layers = nl; kv->kv_dim = kv_dim;
    return kv;
}

static void llama_forward(llama_model* m, kv_cache* kv, int token, int pos, float* logits) {
    int E = m->embed, H = m->n_heads, KV = m->n_kv_heads;
    int HD = m->head_dim, KVD = m->kv_dim, FFN = m->ffn, Q_DIM = m->q_dim;
    float eps = m->rms_eps; int gqa = H / KV;

    float *x = (float*)calloc(E, sizeof(float));
    // ── SPLICE POINT (vision phase: image-placeholder tokens get connector embeds here) ──
    memcpy(x, m->tok_emb + (long)token * E, E * sizeof(float));

    float *xn = (float*)calloc(E, sizeof(float));
    float *q_all = (float*)calloc(Q_DIM, sizeof(float));
    float *k_new = (float*)calloc(KVD, sizeof(float));
    float *v_new = (float*)calloc(KVD, sizeof(float));
    float *attn_out = (float*)calloc(Q_DIM, sizeof(float));
    float *ffn_gate = (float*)calloc(FFN, sizeof(float));
    float *ffn_up = (float*)calloc(FFN, sizeof(float));
    float *ffn_out = (float*)calloc(E, sizeof(float));

    for (int l = 0; l < m->n_layers; l++) {
        rmsnorm(xn, x, m->layers[l].attn_norm, E, eps);
        mm_t(q_all, xn, m->layers[l].wq, 1, E, Q_DIM);
        mm_t(k_new, xn, m->layers[l].wk, 1, E, KVD);
        mm_t(v_new, xn, m->layers[l].wv, 1, E, KVD);
        add_bias(q_all, m->layers[l].q_bias, Q_DIM);
        add_bias(k_new, m->layers[l].k_bias, KVD);
        add_bias(v_new, m->layers[l].v_bias, KVD);
        for (int h = 0; h < H; h++)  rope(q_all + h*HD, pos, HD, m->rope_base);
        for (int h = 0; h < KV; h++) rope(k_new + h*HD, pos, HD, m->rope_base);

        long base = (long)l * kv->max_seq * KVD;
        memcpy(kv->k + base + (long)pos * KVD, k_new, KVD * sizeof(float));
        memcpy(kv->v + base + (long)pos * KVD, v_new, KVD * sizeof(float));

        float scale = 1.0f / sqrtf((float)HD);
        memset(attn_out, 0, Q_DIM * sizeof(float));
        for (int h = 0; h < H; h++) {
            int kv_h = h / gqa;
            float *q = q_all + h * HD;
            float *scores = (float*)calloc(pos + 1, sizeof(float));
            for (int j = 0; j <= pos; j++) {
                float *kj = kv->k + base + (long)j * KVD + kv_h * HD;
                float dot = 0; for (int d = 0; d < HD; d++) dot += q[d] * kj[d];
                scores[j] = dot * scale;
            }
            softmax(scores, pos + 1);
            float *out_h = attn_out + h * HD;
            for (int j = 0; j <= pos; j++) {
                float *vj = kv->v + base + (long)j * KVD + kv_h * HD;
                for (int d = 0; d < HD; d++) out_h[d] += scores[j] * vj[d];
            }
            free(scores);
        }
        float *proj = (float*)calloc(E, sizeof(float));
        mm_t(proj, attn_out, m->layers[l].wo, 1, Q_DIM, E);
        for (int i = 0; i < E; i++) x[i] += proj[i];
        free(proj);

        rmsnorm(xn, x, m->layers[l].ffn_norm, E, eps);
        mm_t(ffn_gate, xn, m->layers[l].wgate, 1, E, FFN);
        mm_t(ffn_up, xn, m->layers[l].wup, 1, E, FFN);
        for (int i = 0; i < FFN; i++) { float g = ffn_gate[i]; ffn_gate[i] = (g / (1.0f + expf(-g))) * ffn_up[i]; }
        mm_t(ffn_out, ffn_gate, m->layers[l].wdown, 1, FFN, E);
        for (int i = 0; i < E; i++) x[i] += ffn_out[i];
    }
    rmsnorm(xn, x, m->out_norm, E, eps);
    float *lm_head = m->has_output_weight ? m->out_weight : m->tok_emb;
    mm_t(logits, xn, lm_head, 1, E, m->vocab);

    free(x); free(xn); free(q_all); free(k_new); free(v_new);
    free(attn_out); free(ffn_gate); free(ffn_up); free(ffn_out);
}

static int argmax(const float *x, int n) {
    int best = 0; float bv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > bv) { bv = x[i]; best = i; }
    return best;
}

static double now_ms(void) { struct timeval tv; gettimeofday(&tv, NULL); return tv.tv_sec*1000.0 + tv.tv_usec/1000.0; }

/* ── main ───────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: %s <model.gguf> [--ids \"1 2 3\"] [--prompt \"text\"] [-n N]\n", argv[0]);
        printf("  --ids    feed raw token ids (isolates decoder; get them via llama-tokenize)\n");
        printf("  --prompt byte-level fallback (NOT real BPE — for quick smoke only)\n");
        return 1;
    }
    // PHASE 2: image preprocessing test (idefics3 tiling + normalize) — no model needed
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--vision-test") && i + 1 < argc) {
            const char *ip = argv[i + 1];
            int nf = 0, S = 0;
            float *fr = smolvlm_preprocess(ip, &nf, &S);
            if (!fr) { fprintf(stderr, "preprocess failed: %s\n", ip); return 1; }
            long n = (long)nf * 3 * S * S;
            float mn = fr[0], mx = fr[0]; double sum = 0;
            for (long k = 0; k < n; k++) { float v = fr[k]; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
            printf("vision-test: %s\n", ip);
            printf("  n_frames=%d  S=%d  shape/frame=[3,%d,%d]  total floats=%ld\n", nf, S, S, S, n);
            printf("  value range: min=%.4f max=%.4f mean=%.4f  (expect [-1,1], mean~0)\n", mn, mx, sum / n);
            printf("  image tokens (64/frame) = %d\n", nf * 64);
            free(fr);
            return 0;
        }
    }

    // PHASE 3: SigLIP vision-tower test — usage: smolvlm --siglip-test <image> <mmproj.gguf>
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--siglip-test") && i + 2 < argc) {
            const char *ip = argv[i + 1], *mmp = argv[i + 2];
            siglip_model *vm = siglip_load(mmp);
            if (!vm) { fprintf(stderr, "siglip_load failed: %s\n", mmp); return 1; }
            int nf = 0, S = 0;
            float *fr = smolvlm_preprocess(ip, &nf, &S);
            if (!fr) { fprintf(stderr, "preprocess failed: %s\n", ip); return 1; }
            int P = siglip_n_patches(vm), D = siglip_hidden(vm);
            float *h = (float*)malloc((long)P * D * sizeof(float));
            if (!h || siglip_encode(vm, fr, h) != 0) { fprintf(stderr, "siglip_encode failed\n"); return 1; }
            long n = (long)P * D, nan = 0; float mn = h[0], mx = h[0]; double sum = 0, csum = 0;
            for (long t = 0; t < n; t++) { float x = h[t];
                if (x != x) nan++; if (x < mn) mn = x; if (x > mx) mx = x; sum += x; csum += (double)x * (t % 97 + 1); }
            double tok0 = 0; for (int j = 0; j < D; j++) tok0 += h[j]; tok0 /= D;
            printf("siglip-test: image=%s mmproj=%s\n", ip, mmp);
            printf("  vision hidden states = [%d, %d]  (frame 0 of %d)\n", P, D, nf);
            printf("  NaN=%ld  min=%.4f  max=%.4f  mean=%.5f  token0_mean=%.5f\n", nan, mn, mx, sum / n, tok0);
            printf("  checksum=%.6f  (determinism: must be identical across runs)\n", csum);
            /* PHASE 4: pixel-shuffle connector -> visual embeddings in text dim */
            int NV = siglip_n_vis_tokens(vm), TD = siglip_text_dim(vm);
            float *vemb = (float*)malloc((long)NV * TD * sizeof(float));
            if (vemb && siglip_connect(vm, h, vemb) == 0) {
                long vn = (long)NV * TD, vnan = 0; float vmn = vemb[0], vmx = vemb[0]; double vsum = 0, vcs = 0;
                for (long t = 0; t < vn; t++) { float x = vemb[t];
                    if (x != x) vnan++; if (x < vmn) vmn = x; if (x > vmx) vmx = x; vsum += x; vcs += (double)x * (t % 97 + 1); }
                printf("  connector: visual embeddings = [%d, %d]  NaN=%ld min=%.4f max=%.4f mean=%.5f\n",
                       NV, TD, vnan, vmn, vmx, vsum / vn);
                printf("  connector checksum=%.6f\n", vcs);
            } else { fprintf(stderr, "siglip_connect failed\n"); return 1; }
            free(vemb); free(h); free(fr); siglip_free(vm);
            return 0;
        }
    }

    const char *model_path = argv[1];
    const char *ids_str = NULL, *prompt = NULL, *text = NULL;
    int max_tokens = 16;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--ids") && i+1 < argc) ids_str = argv[++i];
        else if (!strcmp(argv[i], "--text") && i+1 < argc) text = argv[++i];
        else if (!strcmp(argv[i], "--prompt") && i+1 < argc) prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i+1 < argc) max_tokens = atoi(argv[++i]);
    }

    gguf_file* gf = gguf_open(model_path);
    if (!gf) return 1;
    llama_model* model = llama_load(gf);
    if (!model) { gguf_close(gf); return 1; }

    // GPT-2 byte-level BPE over the GGUF tokenizer (notorch examples/bpe.c, vendored)
    bpe_tokenizer *bpe = bpe_load(model_path);
    if (bpe) printf("bpe: %d tokens loaded\n", bpe_n_vocab(bpe));
    else printf("bpe: load failed (id/byte modes only)\n");

    // build prompt token ids
    int max_seq = 1024;
    int *tokens = (int*)calloc(max_seq, sizeof(int));
    int n_tok = 0;
    if (ids_str) {
        char *buf = strdup(ids_str), *tok = strtok(buf, " ,");
        while (tok && n_tok < max_seq - max_tokens) { tokens[n_tok++] = atoi(tok); tok = strtok(NULL, " ,"); }
        free(buf);
    } else if (text && bpe) {
        n_tok = bpe_encode(bpe, text, tokens, max_seq - max_tokens);   // real GPT-2 BPE
    } else {
        tokens[n_tok++] = 1; // BOS
        const char *p = prompt ? prompt : "Hello";
        for (int i = 0; p[i] && n_tok < max_seq - max_tokens; i++) tokens[n_tok++] = (unsigned char)p[i];
    }

    printf("prompt: %d tokens [", n_tok);
    for (int i = 0; i < n_tok; i++) printf("%d%s", tokens[i], i+1<n_tok?" ":"");
    printf("]\n");

    kv_cache* kv = kv_new(model->n_layers, max_seq, model->kv_dim);
    float *logits = (float*)calloc(model->vocab, sizeof(float));

    double t0 = now_ms();
    for (int i = 0; i < n_tok; i++) llama_forward(model, kv, tokens[i], i, logits);

    printf("\n-- greedy decode (%d tokens) --\n", max_tokens);
    char piece[256], out_text[4096]; out_text[0] = 0; int out_len = 0;
    for (int step = 0; step < max_tokens; step++) {
        int next = argmax(logits, model->vocab);
        piece[0] = 0;
        if (bpe) bpe_decode_token(bpe, next, piece, sizeof(piece));
        printf("  [%d] id=%d  tok=\"%s\"\n", step, next, piece);
        if (next == 2) { printf("  (EOS)\n"); break; }   // SmolLM2 eos=2
        int pl = (int)strlen(piece);
        if (out_len + pl < (int)sizeof(out_text) - 1) { strcpy(out_text + out_len, piece); out_len += pl; }
        int pos = n_tok + step;
        if (pos >= max_seq - 1) break;
        llama_forward(model, kv, next, pos, logits);
    }
    printf("\ngenerated: \"%s\"\n", out_text);
    printf("-- %.0f ms --\n", now_ms() - t0);

    free(logits); free(tokens);
    if (bpe) bpe_free(bpe);
    gguf_close(gf);
    return 0;
}
