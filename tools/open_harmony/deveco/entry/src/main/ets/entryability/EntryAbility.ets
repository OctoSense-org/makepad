import { AbilityConstant, bundleManager, Configuration, ConfigurationConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { display, window } from '@kit.ArkUI';
import { BusinessError, deviceInfo, pasteboard } from '@kit.BasicServicesKit';
import makepad from 'libmakepad.so';

const DOMAIN = 0xCA11;

/** The ability shell: the app is libmakepad.so, which draws into the page's XComponent. */
export default class EntryAbility extends UIAbility {
  private mainWindow: window.Window | undefined = undefined;
  private density: number = 1;

  onCreate(want: Want, _launchParam: AbilityConstant.LaunchParam): void {
    // The Want's `makepad.*` parameters become the app's environment:
    // `aa start ... --ps makepad.HAL_LOG 1` sets MAKEPAD_HAL_LOG=1.
    let launch = '';
    const parameters = want.parameters;
    if (parameters !== undefined) {
      for (const key of Object.keys(parameters)) {
        if (key.startsWith('makepad.')) {
          launch += `${key.substring(8)}=${parameters[key]}\n`;
        }
      }
    }
    let appBuild = '';
    try {
      const info = bundleManager.getBundleInfoForSelfSync(bundleManager.BundleFlag.GET_BUNDLE_INFO_DEFAULT);
      appBuild = `${info.versionName} ${info.versionCode} ${info.updateTime}`;
    } catch (e) {
      hilog.warn(DOMAIN, 'makepad', 'bundle info: %{public}s', JSON.stringify(e));
    }
    try {
      this.density = display.getDefaultDisplaySync().densityPixels;
    } catch (e) {
      hilog.warn(DOMAIN, 'makepad', 'display: %{public}s', JSON.stringify(e));
    }
    const night = this.context.config.colorMode === ConfigurationConstant.ColorMode.COLOR_MODE_DARK;
    makepad.start(this.context.filesDir, this.context.cacheDir, this.density, deviceInfo.osFullName, appBuild, night,
      this.context.resourceManager, (name: string, value: string) => this.call(name, value), launch);
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        hilog.error(DOMAIN, 'makepad', 'loadContent failed: %{public}s', JSON.stringify(err) ?? '');
      }
    });
    windowStage.on('windowStageEvent', (event: window.WindowStageEventType) => {
      if (event === window.WindowStageEventType.ACTIVE) {
        makepad.focus(true);
      } else if (event === window.WindowStageEventType.INACTIVE) {
        makepad.focus(false);
      }
    });
    windowStage.getMainWindow().then((mainWindow) => {
      this.mainWindow = mainWindow;
      // Edge to edge: the app draws under the system bars and is told what they cover.
      mainWindow.setWindowLayoutFullScreen(true).catch((e: BusinessError) => {
        hilog.warn(DOMAIN, 'makepad', 'layout full screen failed: %{public}d', e.code);
      });
      mainWindow.on('avoidAreaChange', () => this.reportAvoidArea());
      mainWindow.on('keyboardHeightChange', (height: number) => makepad.keyboard(height, height > 0));
      this.reportAvoidArea();
    }).catch((e: BusinessError) => {
      hilog.warn(DOMAIN, 'makepad', 'main window unavailable: %{public}d', e.code);
    });
  }

  /** The status bar, the cutout and the navigation indicator, in vp. */
  private reportAvoidArea(): void {
    const mainWindow = this.mainWindow;
    if (mainWindow === undefined) {
      return;
    }
    try {
      const system = mainWindow.getWindowAvoidArea(window.AvoidAreaType.TYPE_SYSTEM);
      const cutout = mainWindow.getWindowAvoidArea(window.AvoidAreaType.TYPE_CUTOUT);
      const indicator = mainWindow.getWindowAvoidArea(window.AvoidAreaType.TYPE_NAVIGATION_INDICATOR);
      const top = Math.max(system.topRect.height, cutout.topRect.height);
      const bottom = Math.max(system.bottomRect.height, indicator.bottomRect.height);
      const left = Math.max(system.leftRect.width, cutout.leftRect.width);
      const right = Math.max(system.rightRect.width, cutout.rightRect.width);
      makepad.avoidArea(top / this.density, right / this.density, bottom / this.density, left / this.density);
    } catch (e) {
      hilog.warn(DOMAIN, 'makepad', 'avoid area query failed: %{public}s', JSON.stringify(e));
    }
  }

  /** What the app asks of the ability (the shell's `napi::call`). */
  private call(name: string, value: string): void {
    const mainWindow = this.mainWindow;
    try {
      if (name === 'terminate') {
        this.context.terminateSelf();
      } else if (name === 'backUnhandled') {
        this.context.moveAbilityToBackground();
      } else if (name === 'copyToClipboard') {
        pasteboard.getSystemPasteboard().setData(pasteboard.createData(pasteboard.MIMETYPE_TEXT_PLAIN, value));
      } else if (mainWindow === undefined) {
        return;
      } else if (name === 'fullScreen') {
        mainWindow.setSpecificSystemBarEnabled('status', value !== '1');
        mainWindow.setSpecificSystemBarEnabled('navigationIndicator', value !== '1');
      } else if (name === 'statusBarDarkIcons') {
        mainWindow.setWindowSystemBarProperties({ statusBarContentColor: value === '1' ? '#000000' : '#FFFFFF' });
      } else if (name === 'keepScreenOn') {
        mainWindow.setWindowKeepScreenOn(value === '1');
      }
    } catch (e) {
      hilog.warn(DOMAIN, 'makepad', '%{public}s failed: %{public}s', name, JSON.stringify(e));
    }
  }

  onConfigurationUpdate(newConfig: Configuration): void {
    makepad.nightMode(newConfig.colorMode === ConfigurationConstant.ColorMode.COLOR_MODE_DARK);
  }

  onForeground(): void {
    makepad.foreground();
  }

  onBackground(): void {
    makepad.background();
  }

  onDestroy(): void {
    makepad.destroy();
  }
};
