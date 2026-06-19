// ui.go — thin immediate-mode widgets over Ebiten (buttons, text, image fit).
// Ebiten has no widget toolkit; these helpers are the whole UI vocabulary.
package main

import (
	"image/color"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"golang.org/x/image/font"
)

// flatly-ish palette (matches the original ttkbootstrap "flatly" theme feel).
var (
	colBG        = color.RGBA{0xf8, 0xf9, 0xfa, 0xff}
	colText      = color.RGBA{0x21, 0x25, 0x29, 0xff}
	colPrimary   = color.RGBA{0x2c, 0x3e, 0x50, 0xff}
	colPrimaryHi = color.RGBA{0x3d, 0x56, 0x70, 0xff}
	colSuccess   = color.RGBA{0x18, 0xbc, 0x9c, 0xff}
	colDanger    = color.RGBA{0xe7, 0x4c, 0x3c, 0xff}
	colSecondary = color.RGBA{0x95, 0xa5, 0xa6, 0xff}
	colInfo      = color.RGBA{0x34, 0x98, 0xdb, 0xff}
	colWhite     = color.RGBA{0xff, 0xff, 0xff, 0xff}
	colGrid      = color.RGBA{0x6c, 0x75, 0x7d, 0xff}
)

func styleColor(style string, hovered bool) color.RGBA {
	switch style {
	case "success":
		return colSuccess
	case "danger":
		return colDanger
	case "secondary":
		return colSecondary
	case "info":
		return colInfo
	default: // primary / outline-primary
		if hovered {
			return colPrimaryHi
		}
		return colPrimary
	}
}

// Button is a rectangular labelled, clickable region.
type Button struct {
	X, Y, W, H float64
	Label      string
	Style      string // primary | success | danger | secondary | info
	Disabled   bool
	hovered    bool
}

func (b *Button) hit(mx, my int) bool {
	x, y := float64(mx), float64(my)
	return x >= b.X && x <= b.X+b.W && y >= b.Y && y <= b.Y+b.H
}

func (b *Button) draw(screen *ebiten.Image, face font.Face) {
	bg := styleColor(b.Style, b.hovered && !b.Disabled)
	if b.Disabled {
		bg = color.RGBA{bg.R/2 + 0x40, bg.G/2 + 0x40, bg.B/2 + 0x40, 0xff}
	}
	vector.DrawFilledRect(screen, float32(b.X), float32(b.Y), float32(b.W), float32(b.H), bg, true)
	drawTextCentered(screen, face, b.Label, b.X, b.Y, b.W, b.H, colWhite)
}

// drawText draws a string with its top-left at (x,y) (converts to baseline).
func drawText(screen *ebiten.Image, face font.Face, s string, x, y float64, clr color.Color) {
	m := face.Metrics()
	text.Draw(screen, s, face, int(x), int(y)+m.Ascent.Round(), clr)
}

func textWidth(face font.Face, s string) int {
	b := text.BoundString(face, s)
	return b.Dx()
}

func textHeight(face font.Face) int {
	m := face.Metrics()
	return m.Ascent.Round() + m.Descent.Round()
}

func drawTextCentered(screen *ebiten.Image, face font.Face, s string, x, y, w, h float64, clr color.Color) {
	tw := textWidth(face, s)
	th := textHeight(face)
	drawText(screen, face, s, x+(w-float64(tw))/2, y+(h-float64(th))/2, clr)
}

// drawImageFit draws img scaled to fit inside the (x,y,w,h) box, centered,
// preserving aspect ratio (never upscales past 1:1). Returns the drawn rect.
func drawImageFit(screen, img *ebiten.Image, x, y, w, h float64) (dx, dy, dw, dh float64) {
	iw, ih := img.Bounds().Dx(), img.Bounds().Dy()
	if iw == 0 || ih == 0 {
		return x, y, 0, 0
	}
	scale := min3(w/float64(iw), h/float64(ih), 1.0)
	dw = float64(iw) * scale
	dh = float64(ih) * scale
	dx = x + (w-dw)/2
	dy = y + (h-dh)/2
	op := &ebiten.DrawImageOptions{}
	op.GeoM.Scale(scale, scale)
	op.GeoM.Translate(dx, dy)
	op.Filter = ebiten.FilterLinear
	screen.DrawImage(img, op)
	return dx, dy, dw, dh
}

func min3(a, b, c float64) float64 {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	return m
}
