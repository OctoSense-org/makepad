// What the HAL's OpenHarmony shell exports (platform/hal/src/backend/ohos/napi.rs).
export const start: (filesDir: string, cacheDir: string, density: number, osVersion: string, appBuild: string,
  nightMode: boolean, resourceManager: object, call: (name: string, value: string, data?: ArrayBuffer) => void,
  launch: string) => void;
export const foreground: () => void;
export const background: () => void;
export const destroy: () => void;
export const focus: (on: boolean) => void;
export const avoidArea: (top: number, right: number, bottom: number, left: number) => void;
export const keyboard: (height: number, open: boolean) => void;
export const backPress: () => void;
export const nightMode: (on: boolean) => void;
export const windowId: (id: number) => void;
// The ability's TLS sockets report to the shell's streams (socket.rs); an id is a decimal string.
export const socketOpened: (id: string, ok: boolean, certificate: ArrayBuffer | undefined) => void;
export const socketData: (id: string, data: ArrayBuffer) => void;
export const socketSent: (id: string, ok: boolean) => void;
export const socketClosed: (id: string) => void;
