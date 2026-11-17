import makepad from 'libmakepad.so';

/** One XComponent, whose surface and touches are the app's (it names the app's library). */
@Entry
@Component
struct Index {
  onBackPress(): boolean {
    makepad.backPress();
    return true;
  }

  build() {
    Stack() {
      XComponent({ id: 'makepad', type: XComponentType.SURFACE, libraryname: 'makepad' })
        .width('100%')
        .height('100%')
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#000000')
  }
}
