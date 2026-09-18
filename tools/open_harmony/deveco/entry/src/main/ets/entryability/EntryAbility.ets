import { AbilityConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window } from '@kit.ArkUI';
import { BusinessError } from '@kit.BasicServicesKit';
import makepad from 'libmakepad.so'
import {ArkGlue} from '../makepad/makepad'

export default class EntryAbility extends UIAbility {
  private studioReview: boolean = false;

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

    // Edge to edge, like the Android activity: the XComponent covers the whole
    // display, the system bars are transparent, and their avoid areas reach the
    // app as safe-area insets so it can keep its own chrome out of them.
    windowStage.getMainWindow().then((mainWindow) => {
      mainWindow.setWindowLayoutFullScreen(true).catch((e: BusinessError) => {
        hilog.warn(0x0000, 'testTag', 'layout full screen failed: %{public}d', e.code);
      });
      // The app draws its own status strip and home indicator (as on Android),
      // so the system's are hidden rather than overlapping them.
      mainWindow.setSpecificSystemBarEnabled('status', false).catch((e: BusinessError) => {
        hilog.warn(0x0000, 'testTag', 'hide status bar failed: %{public}d', e.code);
      });
      mainWindow.setSpecificSystemBarEnabled('navigationIndicator', false).catch((e: BusinessError) => {
        hilog.warn(0x0000, 'testTag', 'hide navigation indicator failed: %{public}d', e.code);
      });
      mainWindow.setWindowSystemBarProperties({
        statusBarColor: '#00000000',
        navigationBarColor: '#00000000',
        statusBarContentColor: '#FFFFFF',
        navigationBarContentColor: '#FFFFFF'
      }).catch((e: BusinessError) => {
        hilog.warn(0x0000, 'testTag', 'system bar properties failed: %{public}d', e.code);
      });
      const report = () => {
        try {
          const system = mainWindow.getWindowAvoidArea(window.AvoidAreaType.TYPE_SYSTEM);
          const nav = mainWindow.getWindowAvoidArea(window.AvoidAreaType.TYPE_NAVIGATION_INDICATOR);
          ArkGlue.getInstance().setAvoidArea(
            Math.max(system.topRect.height, nav.topRect.height),
            Math.max(system.rightRect.width, nav.rightRect.width),
            Math.max(system.bottomRect.height, nav.bottomRect.height),
            Math.max(system.leftRect.width, nav.leftRect.width));
        } catch (e) {
          hilog.warn(0x0000, 'testTag', 'avoid area query failed: %{public}s', JSON.stringify(e));
        }
      };
      mainWindow.on('avoidAreaChange', () => report());
      report();
    }).catch((e: BusinessError) => {
      hilog.warn(0x0000, 'testTag', 'main window unavailable: %{public}d', e.code);
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
