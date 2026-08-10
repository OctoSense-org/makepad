use crate::makepad_draw::*;

// Semantic text roles — the typographic half of "the card declares intent, the
// framework supplies the specifics".
//
// A card used to have to spell out a font stack to get a weight:
//
//     draw_text.text_style: TextStyle{
//         font_family: FontFamily{
//             latin   := FontMember{ res: crate_resource("makepad_widgets:resources/Roboto-Thin.ttf") asc: 0.0 desc: 0.0 }
//             sym     := FontMember{ res: crate_resource("makepad_widgets:resources/NotoSans-Regular.ttf") asc: 0.0 desc: 0.0 }
//             chinese := FontMember{ res: crate_resource("makepad_widgets:resources/LXGWWenKaiRegular.ttf") asc: 0.0 desc: 0.0 }
//         }
//         font_size: 76
//     }
//
// Six lines of backend-specific incantation — resource URIs, makepad's font-chain
// type, its `asc`/`desc` metric fudge — to say "big and thin". It carried three
// separate ways to be silently wrong, and a generated card hit all three:
//
//   * Omit `font_family` entirely and you get the DEFAULT weight, so a 76pt hero
//     renders chunky. Size alone cannot express weight.
//   * Write an explicit `font_family` and it REPLACES the default chain — which is
//     what was carrying CJK and emoji. A Roboto-only hero rendered 上海 and 多云
//     as empty tofu boxes, at the largest text on the screen.
//   * Roboto has no `↑`/`↓`, so a stat line without a NotoSans member drew tofu
//     there instead.
//
// Every role below therefore carries the COMPLETE chain — latin, symbols, CJK,
// colour emoji — so no role can drop coverage, whatever text a card puts in it.
// Under-specifying is impossible: the role IS the specification.
//
// Colour stays with the card, deliberately: it is genuinely per-use (an AQI value
// is coloured by category, a dim row differs from a bright one) whereas weight and
// coverage are not.
//
// `self:` rather than `makepad_widgets:` because these resolve from inside this
// crate — the same form theme_desktop_dark.rs uses.
script_mod! {
    use mod.prelude.widgets_internal.*
    use mod.widgets.Label

    // The hero number: THIN is what makes a large figure elegant instead of
    // shouty. The only role that uses Roboto-Thin.
    mod.widgets.TextHero = Label{
        draw_text.text_style: TextStyle{
            font_family: FontFamily{
                latin   := FontMember{ res: crate_resource("self:resources/Roboto-Thin.ttf") asc: 0.0 desc: 0.0 }
                sym     := FontMember{ res: crate_resource("self:resources/NotoSans-Regular.ttf") asc: 0.0 desc: 0.0 }
                chinese := FontMember{ res: crate_resource("self:resources/LXGWWenKaiRegular.ttf") asc: 0.0 desc: 0.0 }
                emoji   := FontMember{ res: crate_resource("self:resources/NotoColorEmoji.ttf") asc: 0.0 desc: 0.0 }
            }
            font_size: 76
        }
    }

    // A place or screen name.
    mod.widgets.TextTitle = Label{
        draw_text.text_style: TextStyle{
            font_family: FontFamily{
                latin   := FontMember{ res: crate_resource("self:resources/Roboto-Light.ttf") asc: 0.0 desc: 0.0 }
                sym     := FontMember{ res: crate_resource("self:resources/NotoSans-Regular.ttf") asc: 0.0 desc: 0.0 }
                chinese := FontMember{ res: crate_resource("self:resources/LXGWWenKaiRegular.ttf") asc: 0.0 desc: 0.0 }
                emoji   := FontMember{ res: crate_resource("self:resources/NotoColorEmoji.ttf") asc: 0.0 desc: 0.0 }
            }
            font_size: 26
        }
    }

    // A short descriptive line beside an icon.
    mod.widgets.TextBody = Label{
        draw_text.text_style: TextStyle{
            font_family: FontFamily{
                latin   := FontMember{ res: crate_resource("self:resources/Roboto-Light.ttf") asc: 0.0 desc: 0.0 }
                sym     := FontMember{ res: crate_resource("self:resources/NotoSans-Regular.ttf") asc: 0.0 desc: 0.0 }
                chinese := FontMember{ res: crate_resource("self:resources/LXGWWenKaiRegular.ttf") asc: 0.0 desc: 0.0 }
                emoji   := FontMember{ res: crate_resource("self:resources/NotoColorEmoji.ttf") asc: 0.0 desc: 0.0 }
            }
            font_size: 17
        }
    }

    // A secondary figures line — the ↑ ↓ ≈ run lives here, which is why the
    // symbol member is not optional.
    mod.widgets.TextStat = Label{
        draw_text.text_style: TextStyle{
            font_family: FontFamily{
                latin   := FontMember{ res: crate_resource("self:resources/Roboto-Light.ttf") asc: 0.0 desc: 0.0 }
                sym     := FontMember{ res: crate_resource("self:resources/NotoSans-Regular.ttf") asc: 0.0 desc: 0.0 }
                chinese := FontMember{ res: crate_resource("self:resources/LXGWWenKaiRegular.ttf") asc: 0.0 desc: 0.0 }
                emoji   := FontMember{ res: crate_resource("self:resources/NotoColorEmoji.ttf") asc: 0.0 desc: 0.0 }
            }
            font_size: 14
        }
    }

    // A list-row label or value.
    mod.widgets.TextRow = Label{
        draw_text.text_style: TextStyle{
            font_family: FontFamily{
                latin   := FontMember{ res: crate_resource("self:resources/Roboto-Light.ttf") asc: 0.0 desc: 0.0 }
                sym     := FontMember{ res: crate_resource("self:resources/NotoSans-Regular.ttf") asc: 0.0 desc: 0.0 }
                chinese := FontMember{ res: crate_resource("self:resources/LXGWWenKaiRegular.ttf") asc: 0.0 desc: 0.0 }
                emoji   := FontMember{ res: crate_resource("self:resources/NotoColorEmoji.ttf") asc: 0.0 desc: 0.0 }
            }
            font_size: 14
        }
    }

    // A small uppercase caption over a value.
    mod.widgets.TextCaption = Label{
        draw_text.text_style: TextStyle{
            font_family: FontFamily{
                latin   := FontMember{ res: crate_resource("self:resources/Roboto-Light.ttf") asc: 0.0 desc: 0.0 }
                sym     := FontMember{ res: crate_resource("self:resources/NotoSans-Regular.ttf") asc: 0.0 desc: 0.0 }
                chinese := FontMember{ res: crate_resource("self:resources/LXGWWenKaiRegular.ttf") asc: 0.0 desc: 0.0 }
                emoji   := FontMember{ res: crate_resource("self:resources/NotoColorEmoji.ttf") asc: 0.0 desc: 0.0 }
            }
            font_size: 11
        }
    }

    // The prominent figure inside a tile.
    mod.widgets.TextValue = Label{
        draw_text.text_style: TextStyle{
            font_family: FontFamily{
                latin   := FontMember{ res: crate_resource("self:resources/Roboto-Light.ttf") asc: 0.0 desc: 0.0 }
                sym     := FontMember{ res: crate_resource("self:resources/NotoSans-Regular.ttf") asc: 0.0 desc: 0.0 }
                chinese := FontMember{ res: crate_resource("self:resources/LXGWWenKaiRegular.ttf") asc: 0.0 desc: 0.0 }
                emoji   := FontMember{ res: crate_resource("self:resources/NotoColorEmoji.ttf") asc: 0.0 desc: 0.0 }
            }
            font_size: 20
        }
    }
}
