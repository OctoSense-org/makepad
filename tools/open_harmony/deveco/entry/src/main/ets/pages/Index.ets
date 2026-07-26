import {ArkGlue, ComposerState, MakepadXComponentInterface, WebViewSpec, XComponentAttrs} from '../makepad/makepad';

/// The native chat composer, floating over the makepad surface.
///
/// Native for the same reason it is native on Android: a full-screen card
/// (webview or makepad) swallows taps aimed at a makepad-drawn composer, and
/// makepad has no text-input bridge on OHOS — a makepad TextInput would never
/// receive a single character. This overlay owns its own taps and keystrokes
/// and reports them to Rust through the XComponent's napi callbacks.
@Component
struct MakepadComposer {
  @ObjectLink state: ComposerState;
  @State text: string = '';
  controller: TextInputController = new TextInputController();

  private submit(): void {
    const t = this.text.trim();
    if (t.length === 0) {
      return;
    }
    ArkGlue.getInstance().composerSubmitText(t);
    this.text = '';
  }

  build() {
    // Full-bleed container so the pill can sit at the bottom, but with
    // HitTestMode.None so it does NOT eat touches meant for the card or the
    // makepad surface underneath — only its children are interactive.
    Column() {
      if (this.state.expanded) {
        Row() {
          Text('＋')
            .fontSize(22)
            .fontColor('#CFE9E0')
            .width(38)
            .height(38)
            .textAlign(TextAlign.Center)
            .onClick(() => { ArkGlue.getInstance().composerNewAppTapped(); })
          Text('⟳')
            .fontSize(20)
            .fontColor('#CFE9E0')
            .width(38)
            .height(38)
            .textAlign(TextAlign.Center)
            .onClick(() => { ArkGlue.getInstance().composerSwitchTapped(); })
          TextInput({ text: this.text, placeholder: 'Ask anything…', controller: this.controller })
            .layoutWeight(1)
            .fontSize(16)
            .fontColor('#E8EDF2')
            .placeholderColor('#7FA398')
            .backgroundColor(Color.Transparent)
            .borderRadius(0)
            .caretColor('#72E4FF')
            .enterKeyType(EnterKeyType.Send)
            .defaultFocus(true)
            .onChange((v: string) => { this.text = v; })
            .onSubmit(() => { this.submit(); })
          Text('↑')
            .fontSize(20)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.text.trim().length > 0 ? '#0B1F1A' : '#7FA398')
            .width(38)
            .height(38)
            .textAlign(TextAlign.Center)
            .borderRadius(19)
            .backgroundColor(this.text.trim().length > 0 ? '#72E4FF' : '#14322B')
            .onClick(() => { this.submit(); })
        }
        .width('92%')
        .padding({ left: 10, right: 6, top: 6, bottom: 6 })
        .backgroundColor('#E60B4035')
        .borderRadius(24)
        .border({ width: 1.5, color: '#5572E4FF' })
      } else {
        // Collapsed: a single FAB that unfolds the pill.
        Text('＋')
          .fontSize(28)
          .fontColor('#CFE9E0')
          .width(56)
          .height(56)
          .textAlign(TextAlign.Center)
          .borderRadius(28)
          .backgroundColor('#E60B4035')
          .border({ width: 1.5, color: '#5572E4FF' })
          .onClick(() => { ArkGlue.getInstance().composerUnfoldTapped(); })
      }
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.End)
    .alignItems(this.state.expanded ? HorizontalAlign.Center : HorizontalAlign.Start)
    .padding({ left: 16, bottom: 18 })
    .visibility(this.state.visible ? Visibility.Visible : Visibility.Hidden)
    .hitTestBehavior(HitTestMode.None)
  }
}

/// One webview card.
///
/// Split into its own @Component with @ObjectLink so that geometry and
/// visibility changes are observed as *property* mutations on the spec. The
/// alternative — reassigning the whole @State array from ArkGlue on every
/// `webviewUpdate` — happens once per frame and churns the Web component,
/// which resets the loaded document back to `src` (a blank card).
@Component
struct MakepadWebCard {
  @ObjectLink spec: WebViewSpec;

  build() {
    // The controller comes straight from ArkGlue, NOT from the @ObjectLink
    // spec: a controller reached through the observation proxy accepts
    // loadData() and then does nothing at all.
    Web({ src: this.spec.src, controller: ArkGlue.getInstance().controllerFor(this.spec.id) })
      .position({ x: this.spec.x, y: this.spec.y })
      .width(this.spec.w)
      .height(this.spec.h)
      // Hidden rather than removed: keeps the document (and any playing
      // video) alive while the card is scrolled out of view.
      .visibility(this.spec.visible ? Visibility.Visible : Visibility.Hidden)
      .javaScriptAccess(true)
      .domStorageAccess(true)
      .horizontalScrollBarAccess(false)
      .verticalScrollBarAccess(false)
      // Flush on onPageEnd, NOT onControllerAttached. The controller binds
      // before the component's initial `src` navigation completes, so loading
      // the card from onControllerAttached wins the race only to be wiped when
      // the about:blank load lands a few hundred ms later. onPageEnd fires
      // after that settles; later flushes are no-ops (pending is cleared).
      .onPageEnd(() => {
        ArkGlue.getInstance().webviewAttached(this.spec.id);
      })
  }
}

@Entry
@Component
struct Index {
  xComponentContext: MakepadXComponentInterface | undefined = undefined;
  // Webview cards, driven from Rust through ArkGlue. Rendered as an overlay on
  // top of the makepad surface, the same arrangement Android uses (WebView over
  // SurfaceView) — makepad draws the card frame, the Web component draws the
  // document inside it.
  @State webviews: WebViewSpec[] = [];
  xComponentAttrs: XComponentAttrs = {
    id: 'Entry',
    type: XComponentType.SURFACE,
    // Must match the Rust NAPI module name registered by libmakepad.so.
    // 'entry' loads a nonexistent libentry.so: the XComponent then never
    // delivers __NATIVE_XCOMPONENT_OBJ__ to Rust, no surface callbacks fire,
    // and the app shows a blank screen.
    libraryname: 'makepad',
  }

  aboutToAppear(): void {
    ArkGlue.getInstance().onWebViewsChanged = (specs: WebViewSpec[]) => {
      this.webviews = specs;
    };
  }

  build() {
    Stack({ alignContent: Alignment.TopStart }) {
      Row() {
        Column() {
          XComponent(this.xComponentAttrs)
            .focusable(true)
            .onLoad((xComponentContext) => {
              this.xComponentContext = xComponentContext as MakepadXComponentInterface;
              ArkGlue.getInstance().xComponentContext = this.xComponentContext;
              //this.xComponentContext.initMakepad(ArkGlue.getInstance());
            })
        }
        .width('100%')
      }
      .height('100%')

      ForEach(this.webviews, (spec: WebViewSpec) => {
        MakepadWebCard({ spec: spec })
      }, (spec: WebViewSpec) => spec.id)

      // Last child so it floats above the webview cards.
      MakepadComposer({ state: ArkGlue.getInstance().composer })
    }
    .width('100%')
    .height('100%')
  }
}
