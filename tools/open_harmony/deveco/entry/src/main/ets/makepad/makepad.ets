import { window } from '@kit.ArkUI';
import { webview } from '@kit.ArkWeb';
import common from '@ohos.app.ability.common';
import resourceManager from '@ohos.resourceManager';
import abilityAccessCtrl, { Permissions } from '@ohos.abilityAccessCtrl';
import photoAccessHelper from '@ohos.file.photoAccessHelper';
import fileUri from '@ohos.file.fileuri';
import fs from '@ohos.file.fs';

declare type Any = ESObject;

export interface MakepadXComponentInterface{
  onCreate(ark_ts: object):void;
  mountContent(content: object):void;
  handleKeyboardStatus(isOpen:boolean, keyboardHeight:number):void;
  handlePermissionResult(permission: string, granted: boolean, canRetry: boolean): void;
  handleAvoidArea(top: number, right: number, bottom: number, left: number): void;
  takeCaptureToSave(): string;
  webViewList(): Array<string>;
  webViewTakeCommands(): Array<string>;
  webViewAttached(tag: string): void;
  webViewReady(tag: string): void;
  webViewMayNavigate(tag: string, url: string): boolean;
  webViewNavigation(tag: string, url: string, title: string, loading: boolean): void;
  webViewPageError(tag: string, code: number, description: string, url: string): void;
  handleCaptureSaved(path: string, ok: boolean, uri: string): void;
}

export class ArkGlue{
  private static instance : ArkGlue;
  static initInstance(context: common.Context) : ArkGlue {
    if(ArkGlue.instance == null) {
      ArkGlue.instance = new ArkGlue(context);
    }
      return ArkGlue.instance;
  }

  static getInstance() : ArkGlue {
    return ArkGlue.instance;
  }

  context: common.Context;

  resMgr : resourceManager.ResourceManager;
  // The module's directories (haps/<module>/...): the NDK only reports the
  // application's, and the app's data lives here.
  filesDir : string;
  cacheDir : string;
  tempDir : string;
  launchParameters: string = '{}';

  keyboardHeight : number = 1020;
  xComponentContext : MakepadXComponentInterface | undefined = undefined;
  // System bar avoid areas in px, kept here so they can be pushed once the
  // XComponent has loaded (the window reports them before that).
  private avoidArea: number[] = [0, 0, 0, 0];


  constructor(context: common.Context) {
    this.context = context;
    this.resMgr = this.context.resourceManager;
    this.filesDir = this.context.filesDir;
    this.cacheDir = this.context.cacheDir;
    this.tempDir = this.context.tempDir;
  }

  // EntryAbility reports foreground/background so a location session can
  // pause; there is no location bridge on OpenHarmony yet, so this only keeps
  // the ability compiling until one exists.
  setLocationForeground(_foreground: boolean): void {
  }

  // A photo or video the native app wrote into its sandbox: offer it to the
  // system gallery (the user confirms in the system dialog) and copy it over.
  saveLatestCapture(): void {
    const spec = this.xComponentContext?.takeCaptureToSave() ?? '';
    if (spec.length === 0) { return; }
    const parts = spec.split('|');
    const path = parts[0];
    const ext = parts[1] ?? 'jpg';
    const isVideo = (parts[2] ?? 'image') === 'video';
    const helper = photoAccessHelper.getPhotoAccessHelper(this.context);
    const photoType = isVideo ? photoAccessHelper.PhotoType.VIDEO : photoAccessHelper.PhotoType.IMAGE;
    // With ohos.permission.WRITE_IMAGEVIDEO granted (an ACL permission the
    // signing profile allows) the asset is created silently; otherwise the
    // system asks with its save dialog.
    const atManager = abilityAccessCtrl.createAtManager();
    atManager.requestPermissionsFromUser(this.context, ['ohos.permission.WRITE_IMAGEVIDEO']).then((data) => {
      const granted = (data.authResults ?? []).length > 0 && data.authResults[0] === 0;
      if (!granted) { this.saveWithDialog(helper, path, ext, photoType); return; }
      helper.createAsset(photoType, ext).then((uri: string) => {
        this.copyCapture(path, uri);
      }).catch((err: Any) => {
        console.error('createAsset failed, falling back to the dialog: ' + JSON.stringify(err));
        this.saveWithDialog(helper, path, ext, photoType);
      });
    }).catch((err: Any) => {
      console.error('gallery permission request failed: ' + JSON.stringify(err));
      this.saveWithDialog(helper, path, ext, photoType);
    });
  }

  private copyCapture(path: string, uri: string): void {
    try {
      const target = fs.openSync(uri, fs.OpenMode.WRITE_ONLY);
      const source = fs.openSync(path, fs.OpenMode.READ_ONLY);
      fs.copyFileSync(source.fd, target.fd);
      fs.closeSync(source);
      fs.closeSync(target);
      this.xComponentContext?.handleCaptureSaved(path, true, uri);
    } catch (err) {
      console.error('copy capture to gallery failed: ' + JSON.stringify(err));
      this.xComponentContext?.handleCaptureSaved(path, false, '');
    }
  }

  private saveWithDialog(helper: photoAccessHelper.PhotoAccessHelper, path: string, ext: string, photoType: photoAccessHelper.PhotoType): void {
    const config: photoAccessHelper.PhotoCreationConfig = { fileNameExtension: ext, photoType: photoType };
    helper.showAssetsCreationDialog([fileUri.getUriFromPath(path)], [config]).then((uris: Array<string>) => {
      if (uris.length === 0 || uris[0].length === 0) {
        this.xComponentContext?.handleCaptureSaved(path, false, '');
        return;
      }
      this.copyCapture(path, uris[0]);
    }).catch((err: Any) => {
      console.error('gallery dialog failed: ' + JSON.stringify(err));
      this.xComponentContext?.handleCaptureSaved(path, false, '');
    });
  }

  setAvoidArea(top: number, right: number, bottom: number, left: number): void {
    this.avoidArea = [top, right, bottom, left];
    this.pushAvoidArea();
  }

  pushAvoidArea(): void {
    const a = this.avoidArea;
    this.xComponentContext?.handleAvoidArea(a[0], a[1], a[2], a[3]);
    this.listenKeyboardHeightChange();
  }

  // Runtime (user_grant) permissions: Rust calls one of these without
  // arguments (napi values cannot be built off the JS thread) and the answer
  // goes back through handlePermissionResult with the same short name.
  requestCameraPermission(): void {
    this.requestPermission('camera', 'ohos.permission.CAMERA');
  }

  requestMicrophonePermission(): void {
    this.requestPermission('microphone', 'ohos.permission.MICROPHONE');
  }

  requestLocationPermission(): void {
    this.requestPermission('location', 'ohos.permission.APPROXIMATELY_LOCATION');
  }

  private requestPermission(name: string, permission: Permissions): void {
    const atManager = abilityAccessCtrl.createAtManager();
    atManager.requestPermissionsFromUser(this.context, [permission]).then((data) => {
      const results: Array<number> = data.authResults ?? [];
      const dialogs: Array<boolean> = data.dialogShownResults ?? [];
      const granted = results.length > 0 && results[0] === 0;
      const dialogShown = dialogs.length > 0 && dialogs[0] === true;
      console.info('permission ' + permission + ' granted=' + granted + ' dialog=' + dialogShown);
      this.xComponentContext?.handlePermissionResult(name, granted, !granted && dialogShown);
    }).catch((err: Any) => {
      console.error('permission ' + permission + ' failed: ' + JSON.stringify(err));
      this.xComponentContext?.handlePermissionResult(name, false, true);
    });
  }

  // ---- System browsers (platform oh_web.rs) ----
  // The NDK has no web node, so the page declares one `Web` per browser from
  // the list Rust keeps; ArkWeb's NDK then serves the page bridge and script
  // evaluation natively. What it cannot do (load a URL or HTML, history)
  // Rust queues and this runs.
  private webControllers: Map<string, webview.WebviewController> = new Map();
  private webTags: Array<string> = [];

  webController(tag: string): webview.WebviewController {
    let controller = this.webControllers.get(tag);
    if (controller === undefined) {
      // Named with the tag: Rust addresses this webview by it through the NDK.
      controller = new webview.WebviewController(tag);
      this.webControllers.set(tag, controller);
    }
    return controller;
  }

  // Rust calls this when the browsers changed or loads are due.
  syncWebViews(): void {
    const list: Array<string> = this.xComponentContext?.webViewList() ?? [];
    const tags: Array<string> = [];
    const geometry: Record<string, string> = {};
    for (const entry of list) {
      const i = entry.indexOf('|');
      tags.push(entry.substring(0, i));
      geometry[entry.substring(0, i)] = entry.substring(i + 1);
    }
    if (tags.join(',') !== this.webTags.join(',')) {
      for (const tag of this.webTags) {
        if (tags.indexOf(tag) < 0) {
          this.webControllers.delete(tag);
        }
      }
      this.webTags = tags;
      AppStorage.setOrCreate('makepadWebTags', tags);
    }
    AppStorage.setOrCreate('makepadWebGeometry', geometry);
    this.flushWebCommands();
  }

  webAttached(tag: string): void {
    this.xComponentContext?.webViewAttached(tag);
  }

  // The first page (about:blank) finished: loading earlier would race it and
  // be aborted.
  webReady(tag: string): void {
    this.xComponentContext?.webViewReady(tag);
    this.flushWebCommands();
  }

  private flushWebCommands(): void {
    const commands: Array<string> = this.xComponentContext?.webViewTakeCommands() ?? [];
    for (const command of commands) {
      const a = command.indexOf('|');
      const b = command.indexOf('|', a + 1);
      const tag = command.substring(0, a);
      const kind = command.substring(a + 1, b);
      const payload = command.substring(b + 1);
      try {
        const controller = this.webController(tag);
        if (kind === 'url') {
          controller.loadUrl(payload);
        } else if (kind === 'html') {
          const c = payload.indexOf('|');
          controller.loadData(payload.substring(c + 1), 'text/html', 'UTF-8', payload.substring(0, c), ' ');
        } else if (kind === 'history') {
          controller.backOrForward(Number(payload));
        }
      } catch (err) {
        console.error('makepad web ' + kind + ' failed: ' + JSON.stringify(err));
      }
    }
  }

  // The input method itself is native (platform oh_ime.rs, the NDK's
  // libohinputmethod); only the window's keyboard height has no NDK callback.
  private keyboardHeightListening = false;

  private listenKeyboardHeightChange() {
    if (this.keyboardHeightListening) { return; }
    this.keyboardHeightListening = true;
    window.getLastWindow(getContext(this)).then(currentWindow => {
      try {
        currentWindow.on('keyboardHeightChange', (data) => {
          console.info('Keyboard height changed: ' + JSON.stringify(data));
          // A height of 0 is the keyboard going away; the app must hear it,
          // or its view stays panned up by the old height with nothing on
          // screen (the IME switching panels, or dismissed by its own button).
          this.xComponentContext?.handleKeyboardStatus(data != 0, data);
        });
      } catch (exception) {
        console.error(`Failed to enable the listener for keyboard height changes. Cause code: ${exception.code}, message: ${exception.message}`);
      }
    })
  }

}
