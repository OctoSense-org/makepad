// Desktop test for the runhtml web app card pipeline:
// Markdown fence -> web_block template -> WebCard -> native SystemBrowser
// overlay -> set_html (WKWebView on macOS).

pub use makepad_widgets;

use makepad_widgets::*;

app_main!(App);

const TEST_MD: &str = r#"# Web app card test

Below is a runhtml web app card:

```runhtml
<!-- name: youtube-lofi -->
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin: 0; height: 100%; background: #0b0f14; color: #e8edf2;
    font-family: -apple-system, Roboto, sans-serif; }
  .wrap { display: flex; flex-direction: column; height: 100%; padding: 12px; box-sizing: border-box; }
  h1 { font-size: 16px; margin: 0 0 10px 0; }
  .player { flex: 1; border-radius: 12px; overflow: hidden; }
  iframe { width: 100%; height: 100%; border: 0; }
  .row { display: flex; gap: 8px; margin-top: 10px; }
  button { flex: 1; padding: 10px; border-radius: 10px; border: 0; background: #1c2733; color: #e8edf2; font-size: 14px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>lofi hip hop radio — live</h1>
  <div class="player">
    <iframe id="yt" src="https://www.youtube.com/embed/jfKfPfyJRdk?autoplay=1&mute=1&playsinline=1"
      allow="autoplay; encrypted-media; picture-in-picture" allowfullscreen></iframe>
  </div>
  <div class="row">
    <button onclick="document.getElementById('yt').src='https://www.youtube.com/embed/jfKfPfyJRdk?autoplay=1&mute=1&playsinline=1'">Lofi Girl</button>
    <button onclick="document.getElementById('yt').src='https://www.youtube.com/embed/4xDzrJKXOOY?autoplay=1&mute=1&playsinline=1'">Synthwave</button>
  </div>
</div>
</body>
</html>
```

Text after the card renders normally.
"#;

script_mod! {
    use mod.prelude.widgets.*

    startup() do #(App::script_component(vm)){
        ui: Root{
            main_window := Window{
                window.inner_size: vec2(560, 860)
                body +: {
                    View{
                        width: Fill
                        height: Fill
                        flow: Down
                        md := Markdown{
                            width: Fill
                            height: Fill
                            body: #(TEST_MD)
                            web_block := View{
                                width: Fill
                                height: 560
                                web_view := WebCard{
                                    width: Fill
                                    height: Fill
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Script, ScriptHook)]
pub struct App {
    #[live]
    ui: WidgetRef,
}

impl MatchEvent for App {
    fn handle_actions(&mut self, _cx: &mut Cx, _actions: &Actions) {}
}

impl AppMain for App {
    fn script_mod(vm: &mut ScriptVm) -> ScriptValue {
        makepad_widgets::script_mod(vm);
        self::script_mod(vm)
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event) {
        self.match_event(cx, event);
        self.ui.handle_event(cx, event, &mut Scope::empty());
    }
}
