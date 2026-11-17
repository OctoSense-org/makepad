// What the HAL's OpenHarmony shell exports (platform/hal/src/backend/ohos/napi.rs).
export const start: (filesDir: string, cacheDir: string, density: number, osVersion: string, appBuild: string,
  nightMode: boolean, resourceManager: object, call: (name: string, value: string) => void, launch: string) => void;
export const foreground: () => void;
export const background: () => void;
export const destroy: () => void;
export const focus: (on: boolean) => void;
export const avoidArea: (top: number, right: number, bottom: number, left: number) => void;
export const keyboard: (height: number, open: boolean) => void;
export const backPress: () => void;
export const nightMode: (on: boolean) => void;
export const windowId: (id: number) => void;
