import { NodeContent } from '@kit.ArkUI';
import makepad from 'libmakepad.so';
import { ArkGlue, MakepadXComponentInterface } from '../makepad/makepad';

// ArkTS's part in makepad's UI: one ContentSlot, which native code fills
// (node_shim.cpp: the XComponent, its surface, input and scrolling), and one
// `Web` per system browser above it, because the NDK has no web node. Rust
// keeps the browsers (oh_web.rs); their page bridge and script evaluation go
// through ArkWeb's NDK (web_shim.cpp), not through here.
@Entry
@Component
struct Index {
  private content: NodeContent = new NodeContent();
  @StorageLink('makepadWebTags') webTags: Array<string> = [];
  @StorageLink('makepadWebGeometry') webGeometry: Record<string, string> = {};

  aboutToAppear(): void {
    const native = makepad as MakepadXComponentInterface;
    native.mountContent(this.content);
    ArkGlue.getInstance().xComponentContext = native;
    ArkGlue.getInstance().pushAvoidArea();
  }

  // "x|y|w|h|visible" in vp.
  private geometry(tag: string, index: number): number {
    const g = this.webGeometry[tag];
    return g === undefined ? 0 : Number(g.split('|')[index]);
  }

  build() {
    Stack({ alignContent: Alignment.TopStart }) {
      ContentSlot(this.content)
      ForEach(this.webTags, (tag: string) => {
        Web({ src: 'about:blank', controller: ArkGlue.getInstance().webController(tag) })
          .javaScriptAccess(true)
          .domStorageAccess(true)
          .mediaPlayGestureAccess(false)
          .onControllerAttached(() => {
            ArkGlue.getInstance().webAttached(tag);
          })
          // Returning true cancels: a browser opened with `spawn` stays on the
          // documents the app loads.
          .onLoadIntercept((event) => {
            const native = makepad as MakepadXComponentInterface;
            return !native.webViewMayNavigate(tag, event.data.getRequestUrl());
          })
          // about:blank is the placeholder each Web starts on, not a page the
          // app asked for; its end means the browser can take loads.
          .onPageBegin((event) => {
            if (event.url !== 'about:blank') {
              (makepad as MakepadXComponentInterface).webViewNavigation(tag, event.url, '', true);
            }
          })
          .onPageEnd((event) => {
            if (event.url === 'about:blank') {
              ArkGlue.getInstance().webReady(tag);
              return;
            }
            const title = ArkGlue.getInstance().webController(tag).getTitle();
            (makepad as MakepadXComponentInterface).webViewNavigation(tag, event.url, title, false);
          })
          .onErrorReceive((event) => {
            if (event.request.isMainFrame()) {
              (makepad as MakepadXComponentInterface).webViewPageError(tag, event.error.getErrorCode(),
                event.error.getErrorInfo(), event.request.getRequestUrl());
            }
          })
          .position({ x: this.geometry(tag, 0), y: this.geometry(tag, 1) })
          .width(Math.max(1, this.geometry(tag, 2)))
          .height(Math.max(1, this.geometry(tag, 3)))
          .visibility(this.geometry(tag, 4) === 1 ? Visibility.Visible : Visibility.Hidden)
      }, (tag: string) => tag)
    }
    .width('100%')
    .height('100%')
  }
}
