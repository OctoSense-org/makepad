import { AbilityConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';
import { BusinessError } from '@kit.BasicServicesKit';
import makepad from 'libmakepad.so'
import {ArkGlue} from '../makepad/makepad'

export default class EntryAbility extends UIAbility {
  private studioReview: boolean = false;
  private immersive: boolean = false;

  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    let x = ArkGlue.initInstance(this.context);
    const parameters: Record<string, string> = {};
    for (const key of Object.keys(want.parameters ?? {})) {
      const value = want.parameters?.[key];
      if (key.startsWith('makepad.') && typeof value === 'string') {
        parameters[key] = value;
      }
    }
    x.launchParameters = JSON.stringify(parameters);
    this.studioReview = (parameters['makepad.STUDIO_BUILD'] ?? '').length > 0;
    // `--ps makepad.IMMERSIVE 1`: the app owns the whole screen, status bar
    // included — for a shell that draws its own status bar and gesture band.
    this.immersive = parameters['makepad.IMMERSIVE'] === '1';
    makepad.onCreate(x);
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onCreate');
  }

  onDestroy(): void {
    ArkGlue.getInstance().setLocationForeground(false);
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onDestroy');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    // Main window is created, set main page for this ability
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onWindowStageCreate');

    // Studio input reaches the native app rather than the OS input dispatcher,
    // so it does not reset the device's screen timeout during a review session.
    if (this.immersive) {
      windowStage.getMainWindow().then((mainWindow) => {
        mainWindow.setWindowLayoutFullScreen(true).catch((err: BusinessError) => {
          hilog.warn(0x0000, 'testTag', 'Immersive layout failed: %{public}d', err.code);
        });
        mainWindow.setSpecificSystemBarEnabled('status', false).catch((err: BusinessError) => {
          hilog.warn(0x0000, 'testTag', 'Hiding the status bar failed: %{public}d', err.code);
        });
      });
    }
    if (this.studioReview) {
      windowStage.getMainWindow().then((mainWindow) => {
        mainWindow.setWindowKeepScreenOn(true, (err) => {
          if (err.code) {
            hilog.warn(0x0000, 'testTag', 'Studio keep-screen-on failed: %{public}d', err.code);
          }
        });
      });
    }

    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        hilog.error(0x0000, 'testTag', 'Failed to load the content. Cause: %{public}s', JSON.stringify(err) ?? '');
        return;
      }
      hilog.info(0x0000, 'testTag', 'Succeeded in loading the content.');
    });
  }

  onWindowStageDestroy(): void {
    // Main window is destroyed, release UI related resources
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground(): void {
    ArkGlue.getInstance().setLocationForeground(true);
    // Ability has brought to foreground
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onForeground');
  }

  onBackground(): void {
    ArkGlue.getInstance().setLocationForeground(false);
    // Ability has back to background
    hilog.info(0x0000, 'testTag', '%{public}s', 'Ability onBackground');
  }
};
