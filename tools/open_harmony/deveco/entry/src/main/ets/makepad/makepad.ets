import display from '@ohos.display';
import deviceInfo from '@ohos.deviceInfo';
import inputMethod from '@ohos.inputMethod';
import { window } from '@kit.ArkUI';
import common from '@ohos.app.ability.common';
import resourceManager from '@ohos.resourceManager';
import abilityAccessCtrl, { Permissions } from '@ohos.abilityAccessCtrl';
import photoAccessHelper from '@ohos.file.photoAccessHelper';
import fileUri from '@ohos.file.fileuri';
import fs from '@ohos.file.fs';

declare type Any = ESObject;

export interface MakepadXComponentInterface{
  onCreate(ark_ts: object):void;
  handleInsertTextEvent(text:String):void;
  handleDeleteLeftEvent(length: number):void;
  handleKeyboardStatus(isOpen:boolean, keyboardHeight:number):void;
  handlePermissionResult(permission: string, granted: boolean, canRetry: boolean): void;
  handleAvoidArea(top: number, right: number, bottom: number, left: number): void;
  takeCaptureToSave(): string;
  handleCaptureSaved(path: string, ok: boolean, uri: string): void;
}


function getDensity(): number {
  try {
    let displayClass = display.getDefaultDisplaySync();
    console.info('Test densityDPI:' + JSON.stringify(displayClass.densityDPI));
    return displayClass.densityDPI / 160;
  } catch (exception) {
    console.error('Failed to obtain the default display object. Code: ' + JSON.stringify(exception));
    return 3;
  }
}

function getDeviceType(): string {
  let device_type: string = deviceInfo.deviceType;
  if (device_type == "") {
    console.error("deviceInfo.deviceType is empty string!")
  } else {
    console.info("Device type is " + device_type)
  }
  return device_type;
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

  deviceType = getDeviceType();
  osFullName = deviceInfo.osFullName;
  displayDensity = getDensity();
  resMgr : resourceManager.ResourceManager;
  filesDir : string;
  cacheDir : string;
  tempDir : string;
  launchParameters: string = '{}';

  keyboardHeight : number = 1020;
  xComponentContext : MakepadXComponentInterface | undefined = undefined;
  // System bar avoid areas in px, kept here so they can be pushed once the
  // XComponent has loaded (the window reports them before that).
  private avoidArea: number[] = [0, 0, 0, 0];

  private inputMethodController : inputMethod.InputMethodController;
  private textConfig: inputMethod.TextConfig;
  private imcFlag = false;

  constructor(context: common.Context) {
    this.context = context;
    this.resMgr = this.context.resourceManager;
    this.filesDir = this.context.filesDir;
    this.cacheDir = this.context.cacheDir;
    this.tempDir = this.context.tempDir;
    this.inputMethodController = inputMethod.getController();
    this.textConfig = {
      inputAttribute: {
        textInputType: 0,
        enterKeyType: 1
      }
    };
  }

  showKeyBoard(): void {
    this.showTextInput();
  }

  // EntryAbility reports foreground/background so a location session can
  // pause; there is no location bridge on OpenHarmony yet, so this only keeps
  // the ability compiling until one exists.
  setLocationForeground(_foreground: boolean): void {
  }

  hideKeyBoard(): void {
    this.hideTextInput();
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

  private async showTextInput(): Promise<void> {
    this.listenKeyboardHeightChange();
    await this.attach(true);
      if (this.imcFlag != true) {
      this.listenKeyBoardEvent();
    }
    this.inputMethodController.showTextInput().then(() => {
      console.debug("Succeeded in showing softKeyboard");
    }).catch((err: Any) => {
      console.error("Failed to show softKeyboard:" + JSON.stringify(err));
    });
  }

  private async hideTextInput(): Promise<void> {
    this.inputMethodController.hideTextInput().then(() => {
      console.debug("Succeeded in hide softKeyboard");
    }).catch((err: Any) => {
      console.error("Failed to hide softKeyboard:" + JSON.stringify(err));
    })
    this.cancelKeyboardHeightChange();
  }

  private async attach(showKeyboard: boolean): Promise<void> {
    try {
      await this.inputMethodController.attach(showKeyboard, this.textConfig);
    } catch (err) {
      console.error("Failed to attach:" + JSON.stringify(err));
    }
  }

  private listenKeyBoardEvent(): void {
    try {
      this.inputMethodController.on('insertText', (text) => {
      this.xComponentContext?.handleInsertTextEvent(text);
      console.debug("insertText: " + text);
    });
    } catch (err) {
      console.error("Failed to subscribe insertText:" + JSON.stringify(err));
      this.cancelListenKeyBoardEvent();
      return;
    }

    try {
      this.inputMethodController.on('deleteLeft', (length) => {
        this.xComponentContext?.handleDeleteLeftEvent(length);
        console.debug("deleteLeft, length:" + JSON.stringify(length));
      })
    } catch (err) {
      console.error("Failed to subscribe deleteLeft:" + JSON.stringify(err));
      this.cancelListenKeyBoardEvent();
      return;
    }

    try {
      this.inputMethodController.on('sendKeyboardStatus', (state) => {
        if (state == inputMethod.KeyboardStatus.HIDE) {
          this.xComponentContext?.handleKeyboardStatus(false, 0);
          this.inputMethodController.detach();
          this.cancelListenKeyBoardEvent();
        }
        console.debug("sendKeyboardStatus, state:" + JSON.stringify(state));
      })
    } catch (err) {
      console.error("Failed to subscribe sendKeyboardStatus:" + JSON.stringify(err));
      this.cancelListenKeyBoardEvent();
      return;
    }

    console.debug("listenKeyBoardEvent success");
    this.imcFlag = true;
  }

  private cancelListenKeyBoardEvent(): void {
    this.inputMethodController.off('insertText');
    this.inputMethodController.off('deleteLeft');
    this.inputMethodController.off('sendKeyboardStatus');
    this.imcFlag = false;
  }

  private listenKeyboardHeightChange() {
    window.getLastWindow(getContext(this)).then(currentWindow => {
      try {
        currentWindow.on('keyboardHeightChange', (data) => {
          console.info('Succeeded in enabling the listener for keyboard height changes. Data: ' + JSON.stringify(data));
          if (data != 0) {
            this.xComponentContext?.handleKeyboardStatus(true,data);
          }
        });
      } catch (exception) {
        console.error(`Failed to enable the listener for keyboard height changes. Cause code: ${exception.code}, message: ${exception.message}`);
      }
    })
  }

  private cancelKeyboardHeightChange() {
    window.getLastWindow(getContext(this)).then(currentWindow => {
      try {
        currentWindow.off('keyboardHeightChange');
      } catch (exception) {
        console.error(`Failed to enable the listener for keyboard height changes. Cause code: ${exception.code}, message: ${exception.message}`);
      }
    })
  }

}

export interface XComponentAttrs {
  id: string;
  type: number;
  libraryname: string;
}
