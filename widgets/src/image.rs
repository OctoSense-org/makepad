use crate::{
    animator::{Animator, AnimatorAction, AnimatorImpl, Play},
    image_cache::*,
    makepad_derive_widget::*,
    makepad_draw::*,
    makepad_script::ScriptArrayStorage,
    widget::*,
    widget_async::ScriptAsyncResult,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const MAX_SVG_BYTES: usize = 16 * 1024 * 1024;

script_mod! {
    use mod.prelude.widgets_internal.*

    mod.widgets.ImageFit = #(ImageFit::script_api(vm))

    set_type_default() do #(DrawImage::script_shader(vm)){
        ..mod.draw.DrawQuad
        image_texture: texture_2d(float)
        opacity: 1.0
        image_scale: vec2(1.0, 1.0)
        image_pan: vec2(0.0, 0.0)
        fit_scale: vec2(1.0, 1.0)
        fit_pan: vec2(0.0, 0.0)
        async_load: 0.0
        rotation: 0.0
        sample_mode: 0.0
        image_dim_w: 0.0
        image_dim_h: 0.0

        get_color_scale_pan: fn(scale: vec2, pan: vec2) {
            // When image_dim is set, rotate the image rigidly and aspect-correct:
            // map each quad pixel back through the rotation into the image's own
            // pixel rect, so non-square images aren't squished at any angle.
            if self.image_dim_w > 0.0 {
                let angle = self.rotation * 3.141592653589793 / 180.0
                let cos_a = cos(-angle)
                let sin_a = sin(-angle)
                let c = (self.pos - vec2(0.5, 0.5)) * self.rect_size
                let cr = vec2(c.x * cos_a - c.y * sin_a, c.x * sin_a + c.y * cos_a)
                let iuv = cr / vec2(self.image_dim_w, self.image_dim_h) + vec2(0.5, 0.5)
                let uv = iuv * scale + pan
                // Also check iuv, since an animated texture's other frames sit right next to this one.
                if iuv.x < 0.0 || iuv.x > 1.0 || iuv.y < 0.0 || iuv.y > 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
                    return vec4(0.0, 0.0, 0.0, 0.0)
                }
                return self.image_texture.sample_as_bgra(uv)
            }
            let uv = self.pos * scale + pan
            return self.image_texture.sample_as_bgra(uv)
        }

        get_color: fn() {
            return self.get_color_scale_pan(
                self.fit_scale * self.image_scale,
                self.fit_pan * self.image_scale + self.image_pan
            )
        }

        pixel: fn() {
            let color = mix(self.get_color(), #3, self.async_load)
            return Pal.premul(vec4(color.xyz, color.w * self.opacity))
        }
    }

    mod.widgets.ImageBase = #(Image::register_widget(vm))

    mod.widgets.Image = set_type_default() do mod.widgets.ImageBase{
        width: 100
        height: 100
    }
}

#[derive(Script, ScriptHook)]
#[repr(C)]
pub struct DrawImage {
    #[deref]
    draw_super: DrawQuad,
    #[live]
    pub opacity: f32,
    #[live]
    pub image_scale: Vec2f,
    #[live]
    pub image_pan: Vec2f,
    #[live]
    fit_scale: Vec2f,
    #[live]
    fit_pan: Vec2f,
    #[live]
    async_load: f32,
    #[live]
    pub rotation: f32,
    #[live]
    pub sample_mode: f32,
    /// When non-zero, `get_color` rotates the image rigidly (aspect-correct):
    /// the image of this pixel size is rotated by `rotation` and inscribed in the
    /// quad, instead of rotating texture UVs in normalized space (which squishes
    /// non-square images). The image viewer drives these per frame.
    #[live]
    pub image_dim_w: f32,
    #[live]
    pub image_dim_h: f32,
}

#[derive(Copy, Clone, Debug, Default, Script, ScriptHook)]
pub enum ImageAnimation {
    Stop,
    Once,
    #[default]
    Loop,
    Bounce,
    #[live(0.0)]
    Frame(f64),
    #[live(0.0)]
    Factor(f64),
    #[live(60.0)]
    OnceFps(f64),
    #[live(60.0)]
    LoopFps(f64),
    #[live(60.0)]
    BounceFps(f64),
    /// Shows each frame for as long as the image says to, looping forever,
    /// the way browsers play GIFs. It only ticks while the image is on screen.
    Natural,
}

#[derive(Script, ScriptHook, Widget, Animator)]
pub struct Image {
    #[uid]
    uid: WidgetUid,
    #[source]
    source: ScriptObjectRef,
    #[walk]
    pub walk: Walk,
    #[apply_default]
    animator: Animator,
    #[redraw]
    #[live]
    pub draw_bg: DrawImage,
    #[live]
    placeholder_width: u64,
    #[live]
    placeholder_height: u64,
    #[live(1.0)]
    width_scale: f64,
    #[live(ImageAnimation::Natural)]
    animation: ImageAnimation,
    #[rust]
    last_time: Option<f64>,
    #[rust]
    animation_frame: f64,
    /// When a `Natural` animation shows its next frame, on the `AnimationClock`'s timeline.
    /// `None` while it's stopped, e.g., because it went off screen.
    #[rust]
    next_frame_time: Option<f64>,
    #[rust]
    shown_animation: Option<ShownAnimation>,
    #[visible]
    #[live(true)]
    visible: bool,
    #[rust]
    next_frame: NextFrame,
    #[live]
    fit: ImageFit,
    /// Decodes images loaded from data with just enough pixels for the size they're drawn at,
    /// once they're drawn, and again if they're later drawn bigger. So an image that isn't drawn
    /// until it has loaded (e.g., `Media`'s picture) never loads with this on.
    #[live]
    downscale_to_drawn_size: bool,
    /// HTTP/file resource handle for loading image data (set via `http_resource()` or `crate_resource()`)
    #[live]
    src: Option<ScriptHandleRef>,
    #[rust]
    src_loaded: bool,
    #[rust]
    async_image_path: Option<PathBuf>,
    #[rust]
    async_image_size: Option<(usize, usize)>,
    /// The image being loaded or shown, if `downscale_to_drawn_size` might need to decode it again.
    #[rust]
    encoded_image: Option<EncodedImage>,
    #[rust]
    texture: Option<Texture>,
    /// The async-load key that produced `texture`, when it came from a completed
    /// async decode. `None` for textures installed explicitly via `set_texture`
    /// (the current occupant's own content, e.g. a blurhash placeholder), which
    /// must survive the start of an async load; only a texture left behind by an
    /// async load for a different key is stale and gets cleared.
    #[rust]
    texture_async_source: Option<PathBuf>,
    /// `Some` only while showing an SVG (and `texture` is then `None`); lazily
    /// allocated, so non-SVG images pay just a pointer, not a whole `DrawSvg`.
    #[rust]
    draw_svg: Option<Box<DrawSvg>>,
    /// The SVG source currently loaded into `draw_svg`, when the caller supplied it
    /// as shared bytes (see [`Image::load_svg_from_shared_data`]). Holding a share of
    /// the caller's bytes costs a pointer, not a copy, and keeps them alive so their
    /// address stays a valid identity to compare against. `Some` only while `draw_svg`
    /// is, and only for loads that came through the shared-bytes entry point.
    #[rust]
    svg_source: Option<Arc<[u8]>>,
    /// Animation clock (seconds) for animated SVGs, advanced via `next_frame`.
    #[rust]
    svg_time: f64,
}

/// The animated texture an `Image` is drawing, plus the `image_scale` and `image_pan`
/// it had before we pointed them at one frame of it, so a static texture gets them back.
struct ShownAnimation {
    texture_id: TextureId,
    orig_image_scale: Vec2f,
    orig_image_pan: Vec2f,
}

/// An image that's decoded with just enough pixels for the size it's drawn at.
struct EncodedImage {
    image_path: PathBuf,
    /// Loads it with enough pixels to be drawn at the given size.
    load_at_size: Box<dyn Fn(&mut Cx, (usize, usize)) -> Result<AsyncLoadResult, ImageError>>,
    /// The biggest size we've loaded it to be drawn at.
    drawn_size: Option<(usize, usize)>,
}

/// Browsers show a frame that asks for 10ms or less for this long instead,
/// and GIFs made for the web are made with that in mind.
const DEFAULT_FRAME_DELAY_SECS: f64 = 0.1;

/// Frames due within this long after a clock tick change on that tick, since the screen
/// only updates once per display frame anyway. This lets more of them share a repaint.
const ANIMATION_TICK_SLACK_SECS: f64 = 0.008;

/// One clock that steps every `Natural` animation, so all the frames that are due
/// at about the same time change together, in one timer event and one repaint.
#[derive(Default)]
struct AnimationClock {
    /// The timer for the next tick, and when it'll fire.
    next_tick: Timer,
    next_tick_time: f64,
    /// The tick that's being handled right now.
    current_tick: Timer,
}

impl ImageCacheImpl for Image {
    fn get_texture(&self, _id: usize) -> &Option<Texture> {
        &self.texture
    }

    fn set_texture(&mut self, texture: Option<Texture>, _id: usize) {
        self.texture = texture;
        // Keep the invariant that `draw_svg` is `Some` only while showing an SVG.
        self.draw_svg = None;
        self.svg_source = None;
        // The texture is now this widget's content: drop any pending async-load
        // state so the draw path binds it instead of the loading placeholder and
        // a stale decode result for an older key can no longer replace it. It was
        // installed explicitly, so it carries no async source.
        self.async_image_size = None;
        self.async_image_path = None;
        self.texture_async_source = None;
    }

    fn load_image_from_data(
        &mut self,
        cx: &mut Cx,
        data: &[u8],
        id: usize,
    ) -> Result<(), ImageError> {
        if looks_like_svg(data) {
            self.load_svg_from_data(cx, data)
        } else {
            let image = decode_image_from_data(data)?;
            self.set_texture(Some(image.into_new_texture(cx)), id);
            Ok(())
        }
    }
}

impl Image {
    /// Updates layout and aspect fitting without evaluating script. This is
    /// useful for widgets owned by an isolated VM: their typed Rust state can
    /// be changed safely even while the host VM is active.
    pub fn set_walk_and_fit(&mut self, cx: &mut Cx, walk: Walk, fit: ImageFit) {
        if self.walk == walk && self.fit == fit {
            return;
        }
        self.walk = walk;
        self.fit = fit;
        self.redraw(cx);
    }

    pub fn fit(&self) -> ImageFit {
        self.fit
    }

    fn load_from_resource(&mut self, cx: &mut Cx) {
        if self.src_loaded {
            return;
        }
        let Some(ref handle_ref) = self.src else {
            self.src_loaded = true;
            return;
        };
        let handle = handle_ref.as_handle();
        let heap_key = handle_ref.heap_key();
        let data = if let Some(data) = cx.get_resource(heap_key, handle) {
            data
        } else {
            cx.load_script_resource(heap_key, handle);
            match cx.get_resource(heap_key, handle) {
                Some(data) => data,
                None => {
                    let resources = cx.script_data.resources.resources.borrow();
                    if let Some(res) = resources.iter().find(|r| r.has_handle(heap_key, handle)) {
                        if res.is_error() {
                            drop(resources);
                            self.src_loaded = true;
                            return;
                        }
                    } else {
                        self.src_loaded = true;
                    }
                    return; // Not yet loaded (HTTP pending) — retry on next draw
                }
            }
        };
        self.src_loaded = true;
        self.lazy_create_image_cache(cx);
        let path = {
            let resources = cx.script_data.resources.resources.borrow();
            resources
                .iter()
                .find(|r| r.has_handle(heap_key, handle))
                .map(|r| PathBuf::from(&r.abs_path))
                .unwrap_or_else(|| PathBuf::from("http_resource"))
        };
        if let Err(error) = self.load_image_from_data_async(cx, &path, Arc::new((*data).clone())) {
            error!("Image: resource decode failed for {}: {}", path.display(), error);
        }
    }
}

impl Widget for Image {
    fn script_call(
        &mut self,
        vm: &mut ScriptVm,
        method: LiveId,
        args: ScriptValue,
    ) -> ScriptAsyncResult {
        if method == live_id!(set_src) {
            if let Some(args_obj) = args.as_object() {
                let trap = vm.bx.threads.cur().trap.pass();
                let value = vm.bx.heap.vec_value(args_obj, 0, trap);
                if !value.is_err() {
                    if value.is_nil() {
                        vm.with_cx_mut(|cx| {
                            self.src = None;
                            self.src_loaded = false;
                            self.texture = None;
                            self.async_image_path = None;
                            self.async_image_size = None;
                            self.redraw(cx);
                        });
                    } else if let Some(handle) = value.as_handle() {
                        let handle_ref = vm.bx.heap.new_handle_ref(handle);
                        vm.with_cx_mut(|cx| {
                            self.src = Some(handle_ref);
                            self.src_loaded = false;
                            self.texture = None;
                            self.async_image_path = None;
                            self.async_image_size = None;
                            self.redraw(cx);
                        });
                    }
                }
            }
            return ScriptAsyncResult::Return(NIL);
        }
        if method == live_id!(load_image_from_data_async) {
            if let Some(args_obj) = args.as_object() {
                let trap = vm.bx.threads.cur().trap.pass();
                let value = vm.bx.heap.vec_value(args_obj, 0, trap);
                if !value.is_err() {
                    if let Some(data_array) = value.as_array() {
                        if let ScriptArrayStorage::U8(data) = vm.bx.heap.array_storage(data_array) {
                            let path = PathBuf::from(format!(
                                "script_image_data://{}",
                                LiveId::unique().0
                            ));
                            let bytes = Arc::new(data.clone());
                            vm.with_cx_mut(|cx| {
                                let _ = self.load_image_from_data_async(cx, &path, bytes);
                            });
                        }
                    }
                }
            }
            return ScriptAsyncResult::Return(NIL);
        }
        ScriptAsyncResult::MethodNotFound
    }

    fn handle_event(&mut self, cx: &mut Cx, event: &Event, _scope: &mut Scope) {
        if self.animator_handle_event(cx, event).must_redraw() {
            self.draw_bg.redraw(cx);
        }
        if let Event::NetworkResponses(e) = event {
            handle_image_cache_network_responses(cx, e);
        }
        // lets check if we have a post action
        if let Event::Actions(actions) = &event {
            for action in actions {
                if let Some(AsyncImageLoad { image_path, result }) = &action.downcast_ref() {
                    if let Some(result) = result.borrow_mut().take() {
                        // we have a result for the image_cache to load up
                        self.process_async_image_load(cx, image_path, result);
                    }
                    // Only apply a completed decode for the load this widget is still
                    // waiting on; results for other keys belong to a previous occupant
                    // of this (possibly recycled) widget and must be ignored.
                    if self.async_image_size.is_some()
                        && self.async_image_path.as_deref() == Some(image_path.as_path())
                    {
                        let loading_size = self.async_image_size;
                        // see if we can load from cache
                        if !self.load_image_from_cache(cx, image_path, 0) && self.src.is_some() {
                            // Another completion in this Actions batch may
                            // have evicted the texture before this widget got
                            // its turn. Retry from the retained resource bytes.
                            self.src_loaded = false;
                        }
                        self.async_image_size = None;
                        self.async_image_path = None;
                        // Record which async load produced the texture, so a later
                        // load for a different key knows it is stale.
                        self.texture_async_source = Some(image_path.to_path_buf());
                        // A decode with fewer pixels (for another image showing this one) can land
                        // before ours does, so we show that one while we keep waiting for ours.
                        let drawn_size = self.encoded_image.as_ref()
                            .filter(|encoded_image| encoded_image.image_path == *image_path)
                            .map(|encoded_image| encoded_image.drawn_size);
                        let is_waiting_for_more_pixels = match drawn_size {
                            // It hasn't been drawn yet, so any decode of it will do for now.
                            Some(None) => false,
                            // Without `downscale_to_drawn_size`, it wants all of its pixels.
                            drawn_size => {
                                let drawn_size = drawn_size.flatten();
                                is_decoding_image(cx, image_path, drawn_size)
                                    && !self.texture.as_ref().is_some_and(|texture| has_enough_pixels(cx, texture, drawn_size))
                            }
                        };
                        if is_waiting_for_more_pixels {
                            self.async_image_size = loading_size;
                            self.async_image_path = Some(image_path.to_path_buf());
                        } else {
                            self.animator_play(cx, ids!(async_load.off));
                        }
                        self.redraw(cx);
                    }
                }
            }
        }
        if let Some(nf) = self.next_frame.is_event(event) {
            // compute the next frame and patch things up
            if self.draw_svg.is_some() {
                // Animated SVG: advance the clock; the draw step reschedules.
                self.svg_time = nf.time;
                self.redraw(cx);
            } else if let Some(image_texture) = &self.texture {
                let (texture_width, texture_height) = image_texture
                    .get_format(cx)
                    .vec_width_height()
                    .unwrap_or((self.placeholder_width as usize, self.placeholder_height as usize));
                if let Some(animation) = image_texture.animation(cx).clone() {
                    let delta = if let Some(last_time) = &self.last_time {
                        nf.time - last_time
                    } else {
                        0.0
                    };
                    self.last_time = Some(nf.time);
                    let num_frames = animation.num_frames as f64;
                    match self.animation {
                        ImageAnimation::Stop => {}
                        ImageAnimation::Frame(frame) => {
                            self.animation_frame = frame;
                        }
                        ImageAnimation::Factor(pos) => {
                            self.animation_frame = pos * (num_frames - 1.0);
                        }
                        ImageAnimation::Once => {
                            self.animation_frame += 1.0;
                            if self.animation_frame >= num_frames {
                                self.animation_frame = num_frames - 1.0;
                            } else {
                                self.next_frame = cx.new_next_frame();
                            }
                        }
                        ImageAnimation::Loop => {
                            self.animation_frame += 1.0;
                            if self.animation_frame >= num_frames {
                                self.animation_frame = 0.0;
                            }
                            self.next_frame = cx.new_next_frame();
                        }
                        ImageAnimation::Bounce => {
                            self.animation_frame += 1.0;
                            if self.animation_frame >= num_frames * 2.0 {
                                self.animation_frame = 0.0;
                            }
                            self.next_frame = cx.new_next_frame();
                        }
                        ImageAnimation::OnceFps(fps) => {
                            self.animation_frame += delta * fps;
                            if self.animation_frame >= num_frames {
                                self.animation_frame = num_frames - 1.0;
                            } else {
                                self.next_frame = cx.new_next_frame();
                            }
                        }
                        ImageAnimation::LoopFps(fps) => {
                            self.animation_frame += delta * fps;
                            if self.animation_frame >= num_frames {
                                self.animation_frame = 0.0;
                            }
                            self.next_frame = cx.new_next_frame();
                        }
                        ImageAnimation::BounceFps(fps) => {
                            self.animation_frame += delta * fps;
                            if self.animation_frame >= num_frames * 2.0 {
                                self.animation_frame = 0.0;
                            }
                            self.next_frame = cx.new_next_frame();
                        }
                        // Natural animations step on the `AnimationClock` instead.
                        ImageAnimation::Natural => {}
                    }
                    // alright now lets turn animation_frame into the right image_pan
                    let last_pan = self.draw_bg.image_pan;

                    let frame = if self.animation_frame >= num_frames {
                        num_frames * 2.0 - 1.0 - self.animation_frame
                    } else {
                        self.animation_frame
                    } as usize;

                    self.draw_bg.image_pan = get_frame_pan(
                        frame,
                        (animation.width, animation.height),
                        (texture_width, texture_height),
                    );
                    if self.draw_bg.image_pan != last_pan {
                        // patch it into the area
                        self.draw_bg.update_instance_area_value(cx, ids!(image_pan))
                    }
                }
            }
        }
        if let Some(next_frame_time) = self.next_frame_time {
            if is_animation_tick(cx, event) {
                if next_frame_time <= Cx::monotonic_now() + ANIMATION_TICK_SLACK_SECS {
                    self.show_next_animation_frame(cx, next_frame_time);
                } else {
                    request_animation_tick(cx, next_frame_time);
                }
            }
        }
    }

    fn draw_walk(&mut self, cx: &mut Cx2d, _scope: &mut Scope, walk: Walk) -> DrawStep {
        self.load_from_resource(cx);
        self.draw_walk_image(cx, walk)
    }
}

impl Image {
    fn set_crop_to_fill_transform(
        &mut self,
        source_width: f64,
        source_height: f64,
        target_width: f64,
        target_height: f64,
    ) {
        if !source_width.is_finite()
            || !source_height.is_finite()
            || !target_width.is_finite()
            || !target_height.is_finite()
            || source_width <= 0.0
            || source_height <= 0.0
            || target_width <= 0.0
            || target_height <= 0.0
        {
            self.draw_bg.fit_scale = vec2(1.0, 1.0);
            self.draw_bg.fit_pan = vec2(0.0, 0.0);
            return;
        }

        let source_aspect = source_width / source_height;
        let target_aspect = target_width / target_height;
        let mut crop_scale = vec2(1.0, 1.0);

        if source_aspect > target_aspect {
            crop_scale.x = (target_aspect / source_aspect) as f32;
        } else {
            crop_scale.y = (source_aspect / target_aspect) as f32;
        }

        let crop_pan = (vec2(1.0, 1.0) - crop_scale) * 0.5;
        self.draw_bg.fit_scale = crop_scale;
        self.draw_bg.fit_pan = crop_pan;
    }

    /// Returns the original size of the image in pixels (not its displayed size).
    ///
    /// Returns `None` if the image has not been loaded into a texture yet.
    pub fn size_in_pixels(&self, cx: &mut Cx) -> Option<(usize, usize)> {
        if let Some(draw_svg) = self.draw_svg.as_ref() {
            return draw_svg
                .svg_size()
                .map(|sz| (sz.x as usize, sz.y as usize));
        }
        let texture = self.texture.as_ref()?;
        if let Some(natural_size) = texture.natural_size(cx) {
            return Some(natural_size);
        }
        // An animated texture holds all of its frames, so its own size isn't the image's size.
        if let Some(animation) = texture.animation(cx) {
            return Some((animation.width, animation.height));
        }
        texture.get_format(cx).vec_width_height()
    }

    /// Shows the next frame of a `Natural` animation, which was due at `frame_time`,
    /// then waits as long as that frame asks.
    ///
    /// This stops once the image is no longer on screen, and drawing it again restarts it.
    fn show_next_animation_frame(&mut self, cx: &mut Cx, frame_time: f64) {
        self.next_frame_time = None;
        let Some(texture) = self.texture.clone() else { return };
        let Some(texture_size) = texture.get_format(cx).vec_width_height() else { return };
        let Some((frame_width, frame_height, num_frames)) = texture.animation(cx).as_ref()
            .map(|animation| (animation.width, animation.height, animation.num_frames))
        else {
            return;
        };
        if num_frames < 2 || !is_area_on_screen(cx, self.draw_bg.area()) {
            return;
        }
        let frame = (self.animation_frame as usize + 1) % num_frames;
        self.animation_frame = frame as f64;
        self.draw_bg.image_pan = get_frame_pan(frame, (frame_width, frame_height), texture_size);
        self.draw_bg.update_instance_area_value(cx, ids!(image_pan));
        let frame_delay = texture.animation(cx).as_ref()
            .map_or(DEFAULT_FRAME_DELAY_SECS, |animation| get_frame_delay(animation, frame));
        // If we fell behind (e.g., the app was busy), get back on the clock's timeline
        // instead of rushing through frames to catch up.
        let now = Cx::monotonic_now();
        let next_frame_time = Some(frame_time + frame_delay)
            .filter(|&time| time >= now)
            .unwrap_or_else(|| get_aligned_frame_time(now, frame_delay));
        self.next_frame_time = Some(next_frame_time);
        request_animation_tick(cx, next_frame_time);
    }

    /// True if a texture has been set on this `Image`.
    pub fn has_texture(&self) -> bool {
        self.texture.is_some()
    }

    /// True if this `Image` currently has displayable content:
    /// either a raster texture or a loaded SVG.
    ///
    /// This distinguishes "the load was accepted" from "there is actually
    /// something to draw". Note that a pending async load does not imply this
    /// is false: content deliberately kept visible while the load decodes
    /// (a `set_texture` placeholder, or a previous load of the same key)
    /// still counts as content.
    pub fn has_content(&self) -> bool {
        self.texture.is_some() || self.draw_svg.is_some()
    }

    /// Loads an SVG into this `Image` by parsing the UTF-8 SVG `data` and drawing
    /// it with makepad's native vector engine instead of a raster texture.
    ///
    /// The `DrawSvg` is allocated on first use, so images that never show an SVG
    /// carry only a null pointer.
    pub fn load_svg_from_data(&mut self, cx: &mut Cx, data: &[u8]) -> Result<(), ImageError> {
        self.parse_and_show_svg(cx, data)
    }

    /// Like [`Image::load_svg_from_data`], but for source the caller holds in shared
    /// bytes, which lets re-loading the very same source be recognized and skipped.
    ///
    /// Prefer this wherever the load is re-issued on every draw (list items are
    /// repopulated per frame): unlike a raster load, this path is synchronous with no
    /// cache behind it, so without the check it re-parses the source and re-builds its
    /// geometry every single frame. The cached raster path gets the same treatment in
    /// `finish_async_load`.
    pub fn load_svg_from_shared_data(
        &mut self,
        cx: &mut Cx,
        data: Arc<[u8]>,
    ) -> Result<(), ImageError> {
        // Identity of the shared bytes is the check: the same allocation is the same
        // drawing, and holding a share of it keeps the address meaningful (nothing
        // else can be freed into it). This costs one pointer comparison, and holding
        // the source costs a refcount rather than a copy of it.
        if self.draw_svg.is_some()
            && self
                .svg_source
                .as_ref()
                .is_some_and(|shown| Arc::ptr_eq(shown, &data))
        {
            return Ok(());
        }
        self.parse_and_show_svg(cx, &data)?;
        self.svg_source = Some(data);
        Ok(())
    }

    fn parse_and_show_svg(&mut self, cx: &mut Cx, data: &[u8]) -> Result<(), ImageError> {
        if data.len() > MAX_SVG_BYTES {
            return Err(ImageError::DataTooLarge {
                bytes: data.len(),
                limit: MAX_SVG_BYTES,
            });
        }
        let svg_str = std::str::from_utf8(data).map_err(|_| ImageError::UnsupportedFormat)?;
        if self.draw_svg.is_none() {
            self.draw_svg = Some(cx.with_vm(|vm| Box::new(DrawSvg::script_new_with_default(vm))));
        }
        if let Some(draw_svg) = self.draw_svg.as_mut() {
            draw_svg.load_from_str(svg_str);
        }
        // Only a caller that shared its bytes leaves an identity behind to skip on;
        // `load_svg_from_shared_data` records it once this has succeeded.
        self.svg_source = None;
        self.texture = None;
        self.texture_async_source = None;
        // The SVG is now this widget's content: a pending raster load no longer
        // applies, and its decode result must not replace the SVG when it lands.
        self.async_image_size = None;
        self.async_image_path = None;
        self.redraw(cx);
        Ok(())
    }

    pub fn draw_walk_image(&mut self, cx: &mut Cx2d, mut walk: Walk) -> DrawStep {
        if !self.visible {
            return DrawStep::done();
        }
        walk = cx.resolve_walk(walk, ResolveAt::BeforeBegin);
        let svg_time = self.svg_time as f32;
        if let Some(draw_svg) = self.draw_svg.as_mut() {
            draw_svg.draw_walk_time(cx, walk, svg_time);
            let animating = draw_svg.has_animations;
            if animating {
                // Keep ticking so SMIL/CSS-animated SVGs advance.
                self.next_frame = cx.new_next_frame();
            }
            return DrawStep::done();
        }
        // alright we get a walk. depending on our aspect ratio
        // we change either nothing, or width or height
        let rect = cx.peek_walk_turtle(walk);
        let dpi = cx.current_dpi_factor();

        // A decode that landed while this wasn't getting events (e.g., in a closed modal) is picked up now.
        let missed_decode = self.async_image_path.as_deref().and_then(|image_path| {
            let drawn_size = self.encoded_image.as_ref()
                .filter(|encoded_image| encoded_image.image_path == image_path)
                .and_then(|encoded_image| encoded_image.drawn_size);
            if is_decoding_image(cx, image_path, drawn_size) {
                return None;
            }
            load_image_from_cache(cx, image_path).map(|texture| (image_path.to_path_buf(), texture))
        });
        if let Some((image_path, texture)) = missed_decode {
            self.set_texture(Some(texture), 0);
            self.finish_async_load(cx, &image_path);
        }
        // A decode of the image that's loading (just with fewer pixels) is drawn like any other,
        // so an animation keeps playing while it waits for a bigger decode.
        let is_showing_loading_image = self.async_image_path.is_some()
            && self.texture.is_some()
            && self.texture_async_source == self.async_image_path;
        let (width, height) = if let Some((w, h)) = self.async_image_size.filter(|_| !is_showing_loading_image) {
            // Still loading. Any texture present here is legitimate current content
            // (the occupant's own placeholder, or a previous load of this same
            // source; begin_async_load already cleared stale ones), so keep showing
            // it. Otherwise bind the empty texture, never whatever a previous
            // occupant left in the draw vars.
            if let Some(image_texture) = &self.texture {
                self.draw_bg.draw_vars.set_texture(0, image_texture);
            } else {
                self.draw_bg.draw_vars.empty_texture(0);
            }
            (w as f64, h as f64)
        } else if let Some(image_texture) = &self.texture {
            self.draw_bg.draw_vars.set_texture(0, image_texture);
            let (width, height) = image_texture
                .get_format(cx)
                .vec_width_height()
                // A FIXED-size render target (e.g. a video convert pass)
                // has real dimensions too — without this it sized as
                // min_* (usually zero) and the picture silently vanished.
                .or_else(|| image_texture.get_format(cx).render_fixed_width_height())
                .unwrap_or((self.placeholder_width as usize, self.placeholder_height as usize));
            let texture_id = image_texture.texture_id();
            // A texture with fewer pixels than its image is still laid out at the image's size.
            let natural_size = image_texture.natural_size(cx);
            let frame_info = image_texture.animation(cx).as_ref()
                .map(|animation| (animation.width, animation.height, animation.num_frames));
            if frame_info.is_none() {
                if let Some(shown) = self.shown_animation.take() {
                    self.draw_bg.image_scale = shown.orig_image_scale;
                    self.draw_bg.image_pan = shown.orig_image_pan;
                    self.next_frame_time = None;
                }
            }
            if let Some((frame_width, frame_height, num_frames)) = frame_info {
                if self.shown_animation.as_ref().is_none_or(|shown| shown.texture_id != texture_id) {
                    // A newly shown animation starts over from its first frame.
                    let (orig_image_scale, orig_image_pan) = match self.shown_animation.take() {
                        Some(shown) => (shown.orig_image_scale, shown.orig_image_pan),
                        None => (self.draw_bg.image_scale, self.draw_bg.image_pan),
                    };
                    self.shown_animation = Some(ShownAnimation { texture_id, orig_image_scale, orig_image_pan });
                    self.animation_frame = 0.0;
                    self.last_time = None;
                    self.draw_bg.image_pan = vec2(0.0, 0.0);
                    self.next_frame_time = None;
                }
                if !matches!(self.animation, ImageAnimation::Natural) {
                    self.next_frame = cx.new_next_frame();
                } else if num_frames > 1 {
                    // Drawing a natural animation (re)starts it, e.g., after it was off screen,
                    // or after something it's in kept the clock's tick from reaching it.
                    let now = Cx::monotonic_now();
                    let next_frame_time = match self.next_frame_time {
                        Some(time) if time + ANIMATION_TICK_SLACK_SECS >= now => time,
                        _ => {
                            let frame_delay = image_texture.animation(cx).as_ref().map_or(
                                DEFAULT_FRAME_DELAY_SECS,
                                |animation| get_frame_delay(animation, self.animation_frame as usize),
                            );
                            get_aligned_frame_time(now, frame_delay)
                        }
                    };
                    self.next_frame_time = Some(next_frame_time);
                    request_animation_tick(cx, next_frame_time);
                }
                // we have an animation. lets compute the scale and zoom for a certain frame
                let scale_x = frame_width as f32 / width as f32;
                let scale_y = frame_height as f32 / height as f32;
                self.draw_bg.image_scale = vec2(scale_x, scale_y);
                let (frame_width, frame_height) = natural_size.unwrap_or((frame_width, frame_height));
                (frame_width as f64, frame_height as f64)
            } else if image_texture.get_format(cx).is_render() {
                // Render targets are stored top-left on EVERY backend now
                // (GL renders offscreen through a Y-inverted projection),
                // so they sample exactly like any other texture — the old
                // unconditional flip here showed them upside down.
                (width as f64 * self.width_scale, height as f64)
            } else {
                let (width, height) = natural_size.unwrap_or((width, height));
                (width as f64 * self.width_scale, height as f64)
            }
        } else {
            self.draw_bg.draw_vars.empty_texture(0);
            (
                self.placeholder_width as f64 / dpi,
                self.placeholder_height as f64 / dpi,
            )
        };

        let aspect = width / height;
        // A Fit height peeks as NaN, so use its effective content-box max
        // (including a Walk-level max) while preserving intrinsic aspect.
        let height_cap = cx.walk_max_height(walk).unwrap_or(f64::INFINITY);
        let avail_height = if rect.size.y.is_nan() {
            height_cap
        } else {
            rect.size.y.min(height_cap)
        };
        self.draw_bg.fit_scale = vec2(1.0, 1.0);
        self.draw_bg.fit_pan = vec2(0.0, 0.0);
        match self.fit {
            ImageFit::Size => {
                walk.width = Size::Fixed(width);
                walk.height = Size::Fixed(height);
            }
            ImageFit::Stretch => {}
            ImageFit::CropToFill => {
                self.set_crop_to_fill_transform(
                    width,
                    height,
                    rect.size.x,
                    avail_height,
                );
            }
            ImageFit::Horizontal => {
                walk.height = Size::Fixed(rect.size.x / aspect);
            }
            ImageFit::Vertical => {
                walk.width = Size::Fixed(avail_height * aspect);
                walk.height = Size::Fixed(avail_height);
            }
            ImageFit::Smallest => {
                let walk_height = rect.size.x / aspect;
                if walk_height > avail_height {
                    walk.width = Size::Fixed(avail_height * aspect);
                    walk.height = Size::Fixed(avail_height);
                } else {
                    walk.height = Size::Fixed(walk_height);
                }
            }
            ImageFit::Biggest => {
                let walk_height = rect.size.x / aspect;
                if walk_height < avail_height {
                    walk.width = Size::Fixed(avail_height * aspect);
                    walk.height = Size::Fixed(avail_height);
                } else {
                    walk.height = Size::Fixed(walk_height);
                }
            }
        }

        self.draw_bg.draw_walk(cx, walk);

        // Load the image this is loading or showing again if it's now drawn bigger
        // than it was ever loaded for, unless it already has (or is getting) enough pixels.
        if !self.downscale_to_drawn_size {
            return DrawStep::done();
        }
        let is_current = self.encoded_image.as_ref()
            .is_some_and(|encoded_image| self.is_loading_or_showing(&encoded_image.image_path));
        if !is_current {
            // It now shows something else, e.g., a texture that was set directly.
            self.encoded_image = None;
            return DrawStep::done();
        }
        let (drawn_width, drawn_height) = {
            // When it's cropped to fill its rect, the whole image is drawn bigger than that rect.
            let rect = self.draw_bg.area().rect(cx);
            (
                (rect.size.x * dpi / self.draw_bg.fit_scale.x as f64).ceil(),
                (rect.size.y * dpi / self.draw_bg.fit_scale.y as f64).ceil(),
            )
        };
        if !(drawn_width >= 1.0 && drawn_height >= 1.0) {
            return DrawStep::done();
        }
        let (drawn_width, drawn_height) = (drawn_width as usize, drawn_height as usize);
        let Some(encoded_image) = self.encoded_image.as_mut() else { return DrawStep::done() };
        let drawn_size = match encoded_image.drawn_size {
            Some((width, height)) if width >= drawn_width && height >= drawn_height => return DrawStep::done(),
            Some((width, height)) => (width.max(drawn_width), height.max(drawn_height)),
            None => (drawn_width, drawn_height),
        };
        encoded_image.drawn_size = Some(drawn_size);
        if self.async_image_path.is_none()
            && self.texture.as_ref().is_some_and(|texture| has_enough_pixels(cx, texture, Some(drawn_size)))
        {
            return DrawStep::done();
        }
        let image_path = encoded_image.image_path.clone();
        match (encoded_image.load_at_size)(cx, drawn_size) {
            Ok(AsyncLoadResult::Loaded) => {
                // The cache had enough pixels already, e.g., from another image showing this one.
                if self.load_image_from_cache(cx, &image_path, 0) {
                    self.finish_async_load(cx, &image_path);
                    cx.redraw_area_in_draw(self.draw_bg.area());
                }
            }
            Ok(AsyncLoadResult::Loading(width, height)) => {
                if self.async_image_path.is_none() {
                    // Keep showing the one with fewer pixels, at the same size, until this one lands.
                    let natural_size = self.texture.as_ref().and_then(|texture| texture.natural_size(cx));
                    self.begin_async_load(cx, &image_path, natural_size.unwrap_or((width, height)));
                }
            }
            Err(_) => self.cancel_async_load(cx),
        }
        DrawStep::done()
    }

    /// Returns whether this image is already loading or showing the image at `image_path`.
    ///
    /// Loading it again then changes nothing, which matters for widgets that re-issue
    /// their loads on every draw (e.g., list items), since a redraw would just loop.
    fn is_loading_or_showing(&self, image_path: &Path) -> bool {
        match self.async_image_path.as_deref() {
            Some(loading_path) => loading_path == image_path,
            None => self.texture.is_some() && self.texture_async_source.as_deref() == Some(image_path),
        }
    }

    /// Loads the image at the given `image_path` on disk into this `ImageRef`.
    pub fn load_image_file_by_path_async(
        &mut self,
        cx: &mut Cx,
        image_path: &Path,
    ) -> Result<(), ImageError> {
        if self.is_loading_or_showing(image_path) {
            return Ok(());
        }
        self.lazy_create_image_cache(cx);
        match self.load_image_file_by_path_async_impl(cx, image_path, 0) {
            Ok(AsyncLoadResult::Loading(w, h)) => {
                self.begin_async_load(cx, image_path, (w, h));
            }
            Ok(AsyncLoadResult::Loaded) => {
                self.finish_async_load(cx, image_path);
            }
            Err(_) => {
                self.cancel_async_load(cx);
            }
        }
        Ok(())
    }

    pub fn load_image_from_data_async<D>(
        &mut self,
        cx: &mut Cx,
        image_path: &Path,
        data: Arc<D>,
    ) -> Result<(), ImageError>
    where
        D: AsRef<[u8]> + Send + Sync + ?Sized + 'static,
    {
        if self.is_loading_or_showing(image_path) {
            return Ok(());
        }
        self.lazy_create_image_cache(cx);
        if self.downscale_to_drawn_size {
            // Show any cached decode of it right away, since drawing it
            // then decodes it again if it needs more pixels.
            if self.load_image_from_cache(cx, image_path, 0) {
                self.finish_async_load(cx, image_path);
            } else {
                match image_size_by_data((*data).as_ref(), image_path) {
                    Ok(size) => self.begin_async_load(cx, image_path, size),
                    Err(_) => {
                        self.cancel_async_load(cx);
                        return Ok(());
                    }
                }
            }
            let path = image_path.to_path_buf();
            self.encoded_image = Some(EncodedImage {
                image_path: image_path.to_path_buf(),
                load_at_size: Box::new(move |cx, drawn_size| {
                    load_image_from_data_async_at_size(cx, &path, data.clone(), Some(drawn_size))
                }),
                drawn_size: None,
            });
            return Ok(());
        }
        match self.load_image_from_data_async_impl(cx, image_path, data, 0) {
            Ok(AsyncLoadResult::Loading(w, h)) => {
                self.begin_async_load(cx, image_path, (w, h));
            }
            Ok(AsyncLoadResult::Loaded) => {
                self.finish_async_load(cx, image_path);
            }
            Err(_) => {
                self.cancel_async_load(cx);
            }
        }
        Ok(())
    }

    pub fn load_image_http_by_url_async(
        &mut self,
        cx: &mut Cx,
        url: &str,
    ) -> Result<(), ImageError> {
        if self.is_loading_or_showing(Path::new(url)) {
            return Ok(());
        }
        self.lazy_create_image_cache(cx);
        match self.load_image_http_by_url_async_impl(cx, url, 0) {
            Ok(AsyncLoadResult::Loading(w, h)) => {
                self.begin_async_load(cx, Path::new(url), (w, h));
            }
            Ok(AsyncLoadResult::Loaded) => {
                self.finish_async_load(cx, Path::new(url));
            }
            Err(_) => {
                self.cancel_async_load(cx);
            }
        }
        Ok(())
    }

    /// Records `image_path` as this widget's one pending async load, replacing any
    /// previous request so a decode finishing for an older key is never applied.
    fn begin_async_load(&mut self, cx: &mut Cx, image_path: &Path, size: (usize, usize)) {
        // A texture left behind by an async load for a different key belongs to a
        // previous occupant of this (possibly recycled) widget and must not stay
        // visible while the new source decodes. A texture installed via
        // `set_texture` is the current occupant's own content (e.g. a blurhash
        // placeholder) and stays visible until the decode lands, as does the
        // result of a previous load of this same key.
        let texture_is_stale = self
            .texture_async_source
            .as_deref()
            .is_some_and(|source| source != image_path);
        if texture_is_stale {
            self.texture = None;
            self.texture_async_source = None;
        }
        // An SVG is always different content from an incoming raster load.
        self.draw_svg = None;
        self.svg_source = None;
        self.async_image_size = Some(size);
        self.async_image_path = Some(image_path.into());
        self.animator_play(cx, ids!(async_load.on));
        self.redraw(cx);
    }

    /// The requested image was already cached and its texture has been set: clear
    /// the pending-load state so the draw path binds the texture directly.
    fn finish_async_load(&mut self, cx: &mut Cx, image_path: &Path) {
        self.async_image_size = None;
        self.async_image_path = None;
        // Record which load produced the texture, just like the decode-completion
        // path does. Without this, a cache-hit texture has no provenance and a
        // later load for a different key would wrongly keep it visible on a
        // recycled widget while the new source decodes.
        self.texture_async_source = Some(image_path.to_path_buf());
        self.animator_play(cx, ids!(async_load.off));
        self.redraw(cx);
    }

    /// A failed load leaves the widget's intended content unknown: clear both the
    /// pending request (so a decode finishing for a previous key is never applied)
    /// and any displayed content, which may belong to a previous occupant of a
    /// recycled widget. Blank is strictly safer than someone else's image.
    fn cancel_async_load(&mut self, cx: &mut Cx) {
        self.async_image_size = None;
        self.async_image_path = None;
        self.texture = None;
        self.texture_async_source = None;
        self.draw_svg = None;
        self.svg_source = None;
        self.animator_play(cx, ids!(async_load.off));
        self.redraw(cx);
    }
}

/// Returns the `image_pan` that shows the given frame of an animated texture,
/// which holds its frames left to right in rows.
fn get_frame_pan(frame: usize, frame_size: (usize, usize), texture_size: (usize, usize)) -> Vec2f {
    let columns = (texture_size.0 / frame_size.0.max(1)).max(1);
    vec2(
        ((frame % columns) * frame_size.0) as f32 / texture_size.0 as f32,
        ((frame / columns) * frame_size.1) as f32 / texture_size.1 as f32,
    )
}

/// Returns when a frame shown from `now` should change: the first multiple of its delay
/// that's at least a delay away, so animations with the same frame delay change together.
fn get_aligned_frame_time(now: f64, frame_delay: f64) -> f64 {
    ((now + frame_delay) / frame_delay).ceil() * frame_delay
}

/// Asks the shared `AnimationClock` to tick at the given time, unless it'll already tick by then.
fn request_animation_tick(cx: &mut Cx, time: f64) {
    if !cx.has_global::<AnimationClock>() {
        cx.set_global(AnimationClock::default());
    }
    let now = Cx::monotonic_now();
    let clock = cx.get_global::<AnimationClock>();
    // A tick that's past due is never trusted, since its event may have gone unhandled.
    if !clock.next_tick.is_empty() && clock.next_tick_time <= time && clock.next_tick_time >= now {
        return;
    }
    let later_tick = std::mem::replace(&mut clock.next_tick, Timer::empty());
    cx.stop_timer(later_tick);
    let next_tick = cx.start_timeout((time - now).max(0.0));
    let clock = cx.get_global::<AnimationClock>();
    clock.next_tick = next_tick;
    clock.next_tick_time = time;
}

/// Returns whether the given event is a tick of the shared `AnimationClock`.
fn is_animation_tick(cx: &mut Cx, event: &Event) -> bool {
    let Event::Timer(timer_event) = event else { return false };
    if !cx.has_global::<AnimationClock>() {
        return false;
    }
    let clock = cx.get_global::<AnimationClock>();
    // The first image to see a tick frees the clock up to be scheduled again.
    if !clock.next_tick.is_empty() && timer_event.timer_id == clock.next_tick.0 {
        clock.current_tick = std::mem::replace(&mut clock.next_tick, Timer::empty());
    }
    !clock.current_tick.is_empty() && timer_event.timer_id == clock.current_tick.0
}

/// Returns how long the given frame of an animation should be shown, in seconds.
fn get_frame_delay(animation: &TextureAnimation, frame: usize) -> f64 {
    animation.frame_delays.get(frame)
        .copied()
        .filter(|&delay| delay > 0.0101)
        .unwrap_or(DEFAULT_FRAME_DELAY_SECS)
}

/// Returns whether the given area is on screen right now, unlike one in a hidden
/// dock tab or one whose draw list has since been redrawn without it.
fn is_area_on_screen(cx: &Cx, area: Area) -> bool {
    area.is_valid(cx) && area.draw_list_id()
        .and_then(|draw_list_id| cx.draw_lists[draw_list_id].draw_pass_id)
        .is_some_and(|pass_id| !cx.pass_attachment_is_stale(pass_id)
            && area.is_attached(cx, &cx.attached_draw_lists(pass_id)))
}

pub enum AsyncLoad {
    Yes,
    No,
}

impl ImageRef {
    /// See [`Image::set_walk_and_fit`].
    pub fn set_walk_and_fit(&self, cx: &mut Cx, walk: Walk, fit: ImageFit) {
        if let Some(mut inner) = self.borrow_mut() {
            inner.set_walk_and_fit(cx, walk, fit);
        }
    }

    /// Loads the image at the given `image_path` resource into this `ImageRef`.
    pub fn load_image_dep_by_path(&self, cx: &mut Cx, image_path: &str) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.lazy_create_image_cache(cx);
            inner.load_image_dep_by_path(cx, image_path, 0)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    /// Loads the image at the given `image_path` on disk into this `ImageRef`.
    pub fn load_image_file_by_path(
        &self,
        cx: &mut Cx,
        image_path: &Path,
    ) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.lazy_create_image_cache(cx);
            inner.load_image_file_by_path(cx, image_path, 0)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    /// Loads the image at the given `image_path` on disk into this `ImageRef`.
    pub fn load_image_file_by_path_async(
        &self,
        cx: &mut Cx,
        image_path: &Path,
    ) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            return inner.load_image_file_by_path_async(cx, image_path);
        }
        Ok(())
    }

    /// Loads the image at the given `image_path` on disk into this `ImageRef`.
    pub fn load_image_from_data_async<D>(
        &self,
        cx: &mut Cx,
        image_path: &Path,
        data: Arc<D>,
    ) -> Result<(), ImageError>
    where
        D: AsRef<[u8]> + Send + Sync + ?Sized + 'static,
    {
        if let Some(mut inner) = self.borrow_mut() {
            return inner.load_image_from_data_async(cx, image_path, data);
        }
        Ok(())
    }

    /// Loads an image from a URL using platform HTTP + async decode.
    pub fn load_image_http_by_url_async(&self, cx: &mut Cx, url: &str) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            return inner.load_image_http_by_url_async(cx, url);
        }
        Ok(())
    }

    /// Loads a JPEG into this `ImageRef` by decoding the given encoded JPEG `data`.
    pub fn load_jpg_from_data(&self, cx: &mut Cx, data: &[u8]) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.lazy_create_image_cache(cx);
            ImageCacheImpl::load_jpg_from_data(&mut *inner, cx, data, 0)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    /// Loads a PNG into this `ImageRef` by decoding the given encoded PNG `data`.
    pub fn load_png_from_data(&self, cx: &mut Cx, data: &[u8]) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.lazy_create_image_cache(cx);
            ImageCacheImpl::load_png_from_data(&mut *inner, cx, data, 0)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    /// Loads a BMP into this `ImageRef` by decoding the given encoded BMP `data`.
    pub fn load_bmp_from_data(&self, cx: &mut Cx, data: &[u8]) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.lazy_create_image_cache(cx);
            ImageCacheImpl::load_bmp_from_data(&mut *inner, cx, data, 0)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    /// Loads a QOI into this `ImageRef` by decoding the given encoded QOI `data`.
    pub fn load_qoi_from_data(&self, cx: &mut Cx, data: &[u8]) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.lazy_create_image_cache(cx);
            ImageCacheImpl::load_qoi_from_data(&mut *inner, cx, data, 0)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    /// Loads an ICO into this `ImageRef` by decoding the given encoded ICO `data`.
    pub fn load_ico_from_data(&self, cx: &mut Cx, data: &[u8]) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.lazy_create_image_cache(cx);
            ImageCacheImpl::load_ico_from_data(&mut *inner, cx, data, 0)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    /// Loads a GIF into this `ImageRef` by decoding the given encoded GIF `data`.
    pub fn load_gif_from_data(&self, cx: &mut Cx, data: &[u8]) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.lazy_create_image_cache(cx);
            ImageCacheImpl::load_gif_from_data(&mut *inner, cx, data, 0)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    /// Loads a WebP into this `ImageRef` by decoding the given encoded WebP `data`.
    pub fn load_webp_from_data(&self, cx: &mut Cx, data: &[u8]) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.lazy_create_image_cache(cx);
            ImageCacheImpl::load_webp_from_data(&mut *inner, cx, data, 0)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    /// Loads an image into this `ImageRef` by decoding the given encoded `data`,
    /// auto-detecting any image format that makepad supports (including SVG).
    pub fn load_image_from_data(&self, cx: &mut Cx, data: &[u8]) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.lazy_create_image_cache(cx);
            ImageCacheImpl::load_image_from_data(&mut *inner, cx, data, 0)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    /// Loads an SVG into this `ImageRef` by rendering the given UTF-8 SVG `data`
    /// with makepad's native vector engine.
    pub fn load_svg_from_data(&self, cx: &mut Cx, data: &[u8]) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.load_svg_from_data(cx, data)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    /// See [`Image::load_svg_from_shared_data`]: the same, but re-loading the very
    /// same source is recognized and skipped, so a caller that re-issues its load on
    /// every draw doesn't re-parse the SVG every frame.
    pub fn load_svg_from_shared_data(
        &self,
        cx: &mut Cx,
        data: Arc<[u8]>,
    ) -> Result<(), ImageError> {
        if let Some(mut inner) = self.borrow_mut() {
            inner.load_svg_from_shared_data(cx, data)
        } else {
            Ok(()) // preserving existing behavior of silent failures.
        }
    }

    pub fn set_texture(&self, cx: &mut Cx, texture: Option<Texture>) {
        if let Some(mut inner) = self.borrow_mut() {
            // Route through the trait impl so the content invariants (draw_svg,
            // async-load state, texture provenance) live in one place.
            ImageCacheImpl::set_texture(&mut *inner, texture, 0);
            if cx.in_draw_event() {
                inner.redraw(cx);
            }
        }
    }

    pub fn set_uniform(&self, cx: &Cx, uniform: LiveId, value: &[f32]) {
        if let Some(mut inner) = self.borrow_mut() {
            inner.draw_bg.set_uniform(cx, uniform, value);
        }
    }

    /// See [`Image::size_in_pixels()`].
    pub fn size_in_pixels(&self, cx: &mut Cx) -> Option<(usize, usize)> {
        if let Some(inner) = self.borrow() {
            inner.size_in_pixels(cx)
        } else {
            None
        }
    }

    /// See [`Image::has_texture()`].
    pub fn has_texture(&self) -> bool {
        if let Some(inner) = self.borrow() {
            inner.has_texture()
        } else {
            false
        }
    }

    /// See [`Image::has_content()`].
    pub fn has_content(&self) -> bool {
        if let Some(inner) = self.borrow() {
            inner.has_content()
        } else {
            false
        }
    }
}

#[cfg(test)]
mod flattened_walk_collision_tests {
    use super::*;

    #[test]
    fn image_exposes_walk_bounds_and_distinct_placeholder_dimensions() {
        let mut cx = Cx::new(Box::new(|_, _| {}));
        cx.with_vm(|vm| {
            crate::script_mod(vm);
            Image::script_proto(vm);
            let props = &vm
                .bx
                .heap
                .registered_type(Image::script_type_id_static())
                .unwrap()
                .props
                .props;
            for field in [
                live_id!(min_width),
                live_id!(max_width),
                live_id!(min_height),
                live_id!(max_height),
                live_id!(aspect),
                live_id!(placeholder_width),
                live_id!(placeholder_height),
            ] {
                assert!(props.contains_key(&field), "missing flattened/reflected field {field:?}");
            }
        });
    }
}
