use {
    super::{
        geom::{Point, Size, Transform},
        glyph_outline::{Command, GlyphOutline},
        image::{Bgra, SubimageMut},
    },
    std::fmt,
};

const CHANNEL_R: u8 = 0b001;
const CHANNEL_G: u8 = 0b010;
const CHANNEL_B: u8 = 0b100;

const COLOR_YELLOW: u8 = CHANNEL_R | CHANNEL_G;
const COLOR_MAGENTA: u8 = CHANNEL_R | CHANNEL_B;
const COLOR_CYAN: u8 = CHANNEL_G | CHANNEL_B;
const COLOR_WHITE: u8 = CHANNEL_R | CHANNEL_G | CHANNEL_B;

const EDGE_COLORS: [u8; 3] = [COLOR_CYAN, COLOR_MAGENTA, COLOR_YELLOW];

const CHANNELS: [u8; 3] = [CHANNEL_R, CHANNEL_G, CHANNEL_B];

const QUADRATIC_FLATTEN_STEPS: usize = 64;
const CUBIC_FLATTEN_STEPS: usize = 96;

/// Flattened segments of one edge are culled in runs of at most this many,
/// each with its own bounding box. Shorter runs hug the curve but add boxes to
/// test at every pixel; 32 was fastest on Latin and CJK glyphs (4: 12x, 8: 19x,
/// 16: 26x, 32: 28x, 64: 24x faster than the full scan).
const SEGMENT_RUN_LEN: usize = 32;
/// Slack (in pixels) on the culling bound, far above f32 rounding of these
/// distances, so a segment that could tie the winner is never culled.
const CULL_EPSILON: f32 = 1.0 / 64.0;
/// A flattened segment this short (squared, in pixels) has no distance.
const DEGENERATE_SEGMENT_LEN_SQ: f32 = 0.000_000_1;

pub struct Msdfer {
    settings: Settings,
}

impl Msdfer {
    pub fn new(settings: Settings) -> Self {
        Self { settings }
    }

    pub fn settings(&self) -> Settings {
        self.settings
    }

    pub fn outline_to_msdf(
        &mut self,
        outline: &GlyphOutline,
        dpxs_per_em: f32,
        output: &mut SubimageMut<'_, Bgra>,
    ) {
        self.render(outline, dpxs_per_em, output, true);
    }

    /// With `cull`, a pixel skips each run of segments whose bounding box lies
    /// beyond every nearest distance the run could still improve. Those
    /// segments are strictly farther than the winners, and the remaining ones
    /// keep their order, so the image is identical to scanning every segment
    /// (`cull == false`), which costs pixels x segments: ~10^7 distance
    /// evaluations for one CJK glyph.
    fn render(
        &mut self,
        outline: &GlyphOutline,
        dpxs_per_em: f32,
        output: &mut SubimageMut<'_, Bgra>,
        cull: bool,
    ) {
        let output_size = output.size();
        let inner_size = output_size
            .width
            .checked_sub(self.settings.padding * 2)
            .zip(output_size.height.checked_sub(self.settings.padding * 2))
            .map(|(width, height)| Size::new(width, height))
            .expect("output image must include msdf padding on all sides");

        let mut shape = Shape::from_outline(outline);
        color_shape_edges(&mut shape, self.settings.corner_angle_threshold);
        let transform = outline.rasterize_transform(dpxs_per_em);
        let (segments, runs) = flatten_shape(&shape, transform, inner_size.height);

        for y in 0..output_size.height {
            // Nearest |distance| (R, G, B, overall) at the previous pixel of the row.
            let mut previous: Option<[f32; 4]> = None;
            for x in 0..output_size.width {
                let point = Point::new(
                    x as f32 - self.settings.padding as f32 + 0.5,
                    y as f32 - self.settings.padding as f32 + 0.5,
                );
                let mut selector_r = ChannelSelector::default();
                let mut selector_g = ChannelSelector::default();
                let mut selector_b = ChannelSelector::default();
                let mut min_distance = SignedDistance::inf();

                // Upper bounds on this pixel's nearest distances. A distance
                // moves by at most the one-pixel step, so the previous pixel's
                // bound this one; a row's first pixel takes the closest of the
                // runs' farthest box corners.
                let bound = match (cull, previous) {
                    (false, _) => [f32::INFINITY; 4],
                    (true, Some(previous)) => previous.map(|distance| distance + 1.0 + CULL_EPSILON),
                    (true, None) => far_corner_bounds_sq(&runs, point)
                        .map(|bound_sq| bound_sq.sqrt() + CULL_EPSILON),
                };

                for run in &runs {
                    if cull {
                        let mut limit = bound[3];
                        for channel in 0..3 {
                            if run.color & CHANNELS[channel] != 0 {
                                limit = limit.max(bound[channel]);
                            }
                        }
                        if run.near_distance_sq(point) > limit * limit {
                            continue;
                        }
                    }
                    for segment_index in run.segments.clone() {
                        let segment = &segments[segment_index];
                        let (distance, param) = segment.signed_distance(point);
                        if distance_is_better(distance, min_distance) {
                            min_distance = distance;
                        }
                        if segment.color & CHANNEL_R != 0
                            && distance_is_better(distance, selector_r.min_distance)
                        {
                            selector_r.min_distance = distance;
                            selector_r.near_segment_index = Some(segment_index);
                            selector_r.near_param = param;
                        }
                        if segment.color & CHANNEL_G != 0
                            && distance_is_better(distance, selector_g.min_distance)
                        {
                            selector_g.min_distance = distance;
                            selector_g.near_segment_index = Some(segment_index);
                            selector_g.near_param = param;
                        }
                        if segment.color & CHANNEL_B != 0
                            && distance_is_better(distance, selector_b.min_distance)
                        {
                            selector_b.min_distance = distance;
                            selector_b.near_segment_index = Some(segment_index);
                            selector_b.near_param = param;
                        }
                    }
                }

                previous = Some([
                    selector_r.min_distance.distance.abs(),
                    selector_g.min_distance.distance.abs(),
                    selector_b.min_distance.distance.abs(),
                    min_distance.distance.abs(),
                ]);

                if !min_distance.distance.is_finite() {
                    output[Point::new(x, y)] = Bgra::new(0, 0, 0, 255);
                    continue;
                }

                if let Some(segment_index) = selector_r.near_segment_index {
                    segments[segment_index].distance_to_pseudo_distance(
                        &mut selector_r.min_distance,
                        point,
                        selector_r.near_param,
                    );
                }
                if let Some(segment_index) = selector_g.near_segment_index {
                    segments[segment_index].distance_to_pseudo_distance(
                        &mut selector_g.min_distance,
                        point,
                        selector_g.near_param,
                    );
                }
                if let Some(segment_index) = selector_b.near_segment_index {
                    segments[segment_index].distance_to_pseudo_distance(
                        &mut selector_b.min_distance,
                        point,
                        selector_b.near_param,
                    );
                }

                if !selector_r.min_distance.distance.is_finite() {
                    selector_r.min_distance = min_distance;
                }
                if !selector_g.min_distance.distance.is_finite() {
                    selector_g.min_distance = min_distance;
                }
                if !selector_b.min_distance.distance.is_finite() {
                    selector_b.min_distance = min_distance;
                }

                let r = encode_sdf_distance(
                    selector_r.min_distance.distance,
                    self.settings.radius,
                    self.settings.cutoff,
                )
                .to_bits();
                let g = encode_sdf_distance(
                    selector_g.min_distance.distance,
                    self.settings.radius,
                    self.settings.cutoff,
                )
                .to_bits();
                let b = encode_sdf_distance(
                    selector_b.min_distance.distance,
                    self.settings.radius,
                    self.settings.cutoff,
                )
                .to_bits();
                let a = encode_sdf_distance(
                    min_distance.distance,
                    self.settings.radius,
                    self.settings.cutoff,
                )
                .to_bits();
                output[Point::new(x, y)] = Bgra::new(b, g, r, a);
            }
        }
    }
}

impl fmt::Debug for Msdfer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Msdfer").finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Settings {
    pub padding: usize,
    pub radius: f32,
    pub cutoff: f32,
    pub corner_angle_threshold: f32,
}

#[derive(Clone, Debug)]
struct Shape {
    contours: Vec<Contour>,
}

impl Shape {
    fn from_outline(outline: &GlyphOutline) -> Self {
        let mut contours = Vec::new();
        let mut edges = Vec::new();
        let mut first = None;
        let mut last = None;

        for command in outline.commands().iter().copied() {
            match command {
                Command::MoveTo(p) => {
                    push_contour_if_non_empty(&mut contours, &mut edges, first, last);
                    first = Some(p);
                    last = Some(p);
                }
                Command::LineTo(p1) => {
                    if let Some(p0) = last {
                        edges.push(Edge::linear(p0, p1));
                        last = Some(p1);
                    }
                }
                Command::QuadTo(p1, p2) => {
                    if let Some(p0) = last {
                        edges.push(Edge::quadratic(p0, p1, p2));
                        last = Some(p2);
                    }
                }
                Command::CurveTo(p1, p2, p3) => {
                    if let Some(p0) = last {
                        edges.push(Edge::cubic(p0, p1, p2, p3));
                        last = Some(p3);
                    }
                }
                Command::Close => {
                    push_contour_if_non_empty(&mut contours, &mut edges, first, last);
                    first = None;
                    last = None;
                }
            }
        }

        push_contour_if_non_empty(&mut contours, &mut edges, first, last);
        Self { contours }
    }
}

fn push_contour_if_non_empty(
    contours: &mut Vec<Contour>,
    edges: &mut Vec<Edge>,
    first: Option<Point<f32>>,
    last: Option<Point<f32>>,
) {
    if edges.is_empty() {
        return;
    }
    if let (Some(first), Some(last)) = (first, last) {
        if !is_point_almost_equal(first, last) {
            edges.push(Edge::linear(last, first));
        }
    }
    contours.push(Contour {
        edges: std::mem::take(edges),
    });
}

fn color_shape_edges(shape: &mut Shape, corner_angle_threshold: f32) {
    for contour in &mut shape.contours {
        color_contour_edges(contour, corner_angle_threshold);
    }
}

fn color_contour_edges(contour: &mut Contour, corner_angle_threshold: f32) {
    let edge_count = contour.edges.len();
    if edge_count == 0 {
        return;
    }

    let corners = detect_corners(contour, corner_angle_threshold);
    if !corners.iter().any(|is_corner| *is_corner) {
        for edge in &mut contour.edges {
            edge.color = COLOR_WHITE;
        }
        return;
    }

    for start_color in 0..EDGE_COLORS.len() {
        let colors = colors_from_corners(edge_count, &corners, start_color);
        if is_valid_coloring(&colors, &corners) {
            for (edge, color) in contour.edges.iter_mut().zip(colors.iter().copied()) {
                edge.color = color;
            }
            return;
        }
    }

    let mut colors = (0..edge_count)
        .map(|index| EDGE_COLORS[index % EDGE_COLORS.len()])
        .collect::<Vec<_>>();
    if edge_count > 1 && colors[edge_count - 1] == colors[0] {
        colors[edge_count - 1] = EDGE_COLORS[(edge_count + 1) % EDGE_COLORS.len()];
    }
    for (edge, color) in contour.edges.iter_mut().zip(colors.iter().copied()) {
        edge.color = color;
    }
}

fn detect_corners(contour: &Contour, corner_angle_threshold: f32) -> Vec<bool> {
    let edge_count = contour.edges.len();
    if edge_count == 0 {
        return Vec::new();
    }

    let cross_threshold = corner_angle_threshold
        .clamp(0.0, std::f32::consts::PI)
        .sin();
    let mut corners = vec![false; edge_count];
    for (edge_index, is_corner_flag) in corners.iter_mut().enumerate().take(edge_count) {
        let prev_direction = contour.edges[edge_index].tangent_end().normalized();
        let next_direction = contour.edges[(edge_index + 1) % edge_count]
            .tangent_start()
            .normalized();
        if prev_direction.is_zero() || next_direction.is_zero() {
            *is_corner_flag = true;
            continue;
        }
        *is_corner_flag = is_corner(prev_direction, next_direction, cross_threshold);
    }
    corners
}

fn is_corner(prev_direction: Vec2, next_direction: Vec2, cross_threshold: f32) -> bool {
    prev_direction.cross(next_direction).abs() > cross_threshold
        || prev_direction.dot(next_direction) <= 0.0
}

fn colors_from_corners(edge_count: usize, corners: &[bool], start_color: usize) -> Vec<u8> {
    let mut colors = Vec::with_capacity(edge_count);
    let mut color_index = start_color;
    for is_corner in corners.iter().copied().take(edge_count) {
        colors.push(EDGE_COLORS[color_index]);
        if is_corner {
            color_index = (color_index + 1) % EDGE_COLORS.len();
        }
    }
    colors
}

fn is_valid_coloring(colors: &[u8], corners: &[bool]) -> bool {
    if colors.len() != corners.len() {
        return false;
    }
    for edge_index in 0..colors.len() {
        if corners[edge_index] && colors[edge_index] == colors[(edge_index + 1) % colors.len()] {
            return false;
        }
    }
    true
}

fn flatten_shape(
    shape: &Shape,
    transform: Transform<f32>,
    inner_height: usize,
) -> (Vec<FlatSegment>, Vec<SegmentRun>) {
    let mut segments = Vec::new();
    let mut runs = Vec::new();
    for contour in &shape.contours {
        for edge in &contour.edges {
            let edge_start = segments.len();
            flatten_edge(*edge, transform, inner_height, &mut segments);
            let mut start = edge_start;
            while start < segments.len() {
                let end = (start + SEGMENT_RUN_LEN).min(segments.len());
                runs.push(SegmentRun::new(&segments, start..end, edge.color));
                start = end;
            }
        }
    }
    (segments, runs)
}

/// Consecutive flattened segments of one edge and the box that contains them.
struct SegmentRun {
    segments: std::ops::Range<usize>,
    color: u8,
    min: Point<f32>,
    max: Point<f32>,
    /// Whether a segment of the run has a finite distance, which its box's
    /// farthest corner then bounds. A run of degenerate segments bounds nothing.
    bounds_nearest: bool,
}

impl SegmentRun {
    fn new(segments: &[FlatSegment], range: std::ops::Range<usize>, color: u8) -> Self {
        let mut min = Point::new(f32::INFINITY, f32::INFINITY);
        let mut max = Point::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
        for segment in &segments[range.clone()] {
            for p in [segment.p0, segment.p1] {
                min = Point::new(min.x.min(p.x), min.y.min(p.y));
                max = Point::new(max.x.max(p.x), max.y.max(p.y));
            }
        }
        let bounds_nearest = segments[range.clone()].iter().any(|segment| {
            let v = Vec2::from_points(segment.p0, segment.p1);
            v.dot(v) > DEGENERATE_SEGMENT_LEN_SQ
        });
        Self {
            segments: range,
            color,
            min,
            max,
            bounds_nearest,
        }
    }

    /// Squared distance from `point` to the box: no segment of the run is nearer.
    fn near_distance_sq(&self, point: Point<f32>) -> f32 {
        let x = (self.min.x - point.x).max(point.x - self.max.x).max(0.0);
        let y = (self.min.y - point.y).max(point.y - self.max.y).max(0.0);
        x * x + y * y
    }

    /// Squared distance from `point` to the box's farthest corner: the run's
    /// nearest segment (if it has a distance) is no farther.
    fn far_distance_sq(&self, point: Point<f32>) -> f32 {
        let x = (point.x - self.min.x).abs().max((self.max.x - point.x).abs());
        let y = (point.y - self.min.y).abs().max((self.max.y - point.y).abs());
        x * x + y * y
    }
}

/// Squared upper bounds on the nearest distance per channel (R, G, B) and
/// overall: the closest farthest corner among the runs that can be nearest.
fn far_corner_bounds_sq(runs: &[SegmentRun], point: Point<f32>) -> [f32; 4] {
    let mut bound_sq = [f32::INFINITY; 4];
    for run in runs.iter().filter(|run| run.bounds_nearest) {
        let far_sq = run.far_distance_sq(point);
        bound_sq[3] = bound_sq[3].min(far_sq);
        for channel in 0..3 {
            if run.color & CHANNELS[channel] != 0 {
                bound_sq[channel] = bound_sq[channel].min(far_sq);
            }
        }
    }
    bound_sq
}

fn flatten_edge(
    edge: Edge,
    transform: Transform<f32>,
    inner_height: usize,
    segments: &mut Vec<FlatSegment>,
) {
    match edge.segment {
        EdgeSegment::Linear { p0, p1 } => {
            push_flat_segment(
                transform_point_to_image(p0, transform, inner_height),
                transform_point_to_image(p1, transform, inner_height),
                edge.color,
                segments,
            );
        }
        EdgeSegment::Quadratic { p0, p1, p2 } => {
            let mut prev = transform_point_to_image(p0, transform, inner_height);
            for step in 1..=QUADRATIC_FLATTEN_STEPS {
                let t = step as f32 / QUADRATIC_FLATTEN_STEPS as f32;
                let next = transform_point_to_image(
                    quadratic_point(p0, p1, p2, t),
                    transform,
                    inner_height,
                );
                push_flat_segment(prev, next, edge.color, segments);
                prev = next;
            }
        }
        EdgeSegment::Cubic { p0, p1, p2, p3 } => {
            let mut prev = transform_point_to_image(p0, transform, inner_height);
            for step in 1..=CUBIC_FLATTEN_STEPS {
                let t = step as f32 / CUBIC_FLATTEN_STEPS as f32;
                let next = transform_point_to_image(
                    cubic_point(p0, p1, p2, p3, t),
                    transform,
                    inner_height,
                );
                push_flat_segment(prev, next, edge.color, segments);
                prev = next;
            }
        }
    }
}

fn push_flat_segment(p0: Point<f32>, p1: Point<f32>, color: u8, segments: &mut Vec<FlatSegment>) {
    if is_point_almost_equal(p0, p1) {
        return;
    }
    segments.push(FlatSegment { p0, p1, color });
}

fn transform_point_to_image(
    point: Point<f32>,
    transform: Transform<f32>,
    inner_height: usize,
) -> Point<f32> {
    let p = point.apply_transform(transform);
    Point::new(p.x, inner_height as f32 - 1.0 - p.y)
}

fn encode_sdf_distance(distance: f32, radius: f32, cutoff: f32) -> sdfer::Unorm8 {
    let safe_radius = radius.max(0.0001);
    sdfer::Unorm8::encode(1.0 - (distance / safe_radius + cutoff))
}

fn quadratic_point(p0: Point<f32>, p1: Point<f32>, p2: Point<f32>, t: f32) -> Point<f32> {
    let omt = 1.0 - t;
    Point::new(
        omt * omt * p0.x + 2.0 * omt * t * p1.x + t * t * p2.x,
        omt * omt * p0.y + 2.0 * omt * t * p1.y + t * t * p2.y,
    )
}

fn cubic_point(
    p0: Point<f32>,
    p1: Point<f32>,
    p2: Point<f32>,
    p3: Point<f32>,
    t: f32,
) -> Point<f32> {
    let omt = 1.0 - t;
    let omt2 = omt * omt;
    let t2 = t * t;
    Point::new(
        omt2 * omt * p0.x + 3.0 * omt2 * t * p1.x + 3.0 * omt * t2 * p2.x + t2 * t * p3.x,
        omt2 * omt * p0.y + 3.0 * omt2 * t * p1.y + 3.0 * omt * t2 * p2.y + t2 * t * p3.y,
    )
}

fn is_point_almost_equal(a: Point<f32>, b: Point<f32>) -> bool {
    const EPS: f32 = 0.0001;
    (a.x - b.x).abs() <= EPS && (a.y - b.y).abs() <= EPS
}

#[derive(Clone, Debug)]
struct Contour {
    edges: Vec<Edge>,
}

#[derive(Clone, Copy, Debug)]
struct Edge {
    segment: EdgeSegment,
    color: u8,
}

impl Edge {
    fn linear(p0: Point<f32>, p1: Point<f32>) -> Self {
        Self {
            segment: EdgeSegment::Linear { p0, p1 },
            color: COLOR_WHITE,
        }
    }

    fn quadratic(p0: Point<f32>, p1: Point<f32>, p2: Point<f32>) -> Self {
        Self {
            segment: EdgeSegment::Quadratic { p0, p1, p2 },
            color: COLOR_WHITE,
        }
    }

    fn cubic(p0: Point<f32>, p1: Point<f32>, p2: Point<f32>, p3: Point<f32>) -> Self {
        Self {
            segment: EdgeSegment::Cubic { p0, p1, p2, p3 },
            color: COLOR_WHITE,
        }
    }

    fn tangent_start(self) -> Vec2 {
        match self.segment {
            EdgeSegment::Linear { p0, p1 } => Vec2::from_points(p0, p1),
            EdgeSegment::Quadratic { p0, p1, p2 } => {
                let first = Vec2::from_points(p0, p1);
                if !first.is_zero() {
                    first
                } else {
                    Vec2::from_points(p0, p2)
                }
            }
            EdgeSegment::Cubic { p0, p1, p2, p3 } => {
                let first = Vec2::from_points(p0, p1);
                if !first.is_zero() {
                    first
                } else {
                    let second = Vec2::from_points(p0, p2);
                    if !second.is_zero() {
                        second
                    } else {
                        Vec2::from_points(p0, p3)
                    }
                }
            }
        }
    }

    fn tangent_end(self) -> Vec2 {
        match self.segment {
            EdgeSegment::Linear { p0, p1 } => Vec2::from_points(p0, p1),
            EdgeSegment::Quadratic { p0, p1, p2 } => {
                let first = Vec2::from_points(p1, p2);
                if !first.is_zero() {
                    first
                } else {
                    Vec2::from_points(p0, p2)
                }
            }
            EdgeSegment::Cubic { p0, p1, p2, p3 } => {
                let first = Vec2::from_points(p2, p3);
                if !first.is_zero() {
                    first
                } else {
                    let second = Vec2::from_points(p1, p3);
                    if !second.is_zero() {
                        second
                    } else {
                        Vec2::from_points(p0, p3)
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum EdgeSegment {
    Linear {
        p0: Point<f32>,
        p1: Point<f32>,
    },
    Quadratic {
        p0: Point<f32>,
        p1: Point<f32>,
        p2: Point<f32>,
    },
    Cubic {
        p0: Point<f32>,
        p1: Point<f32>,
        p2: Point<f32>,
        p3: Point<f32>,
    },
}

#[derive(Clone, Copy, Debug)]
struct FlatSegment {
    p0: Point<f32>,
    p1: Point<f32>,
    color: u8,
}

impl FlatSegment {
    fn signed_distance(self, point: Point<f32>) -> (SignedDistance, f32) {
        let segment = Vec2::from_points(self.p0, self.p1);
        let segment_len_sq = segment.dot(segment);
        if segment_len_sq <= DEGENERATE_SEGMENT_LEN_SQ {
            return (SignedDistance::inf(), 0.0);
        }

        let aq = Vec2::from_points(self.p0, point);
        let bq = Vec2::from_points(self.p1, point);
        let param = aq.dot(segment) / segment_len_sq;
        let endpoint = if param > 0.5 { bq } else { aq };
        let endpoint_distance = endpoint.length();
        if param > 0.0 && param < 1.0 {
            let ortho_distance = segment.orthonormal(false).dot(aq);
            if ortho_distance.abs() < endpoint_distance {
                return (SignedDistance::new(ortho_distance, 0.0), param);
            }
        }
        if endpoint_distance <= 0.000_000_1 {
            return (SignedDistance::new(0.0, 0.0), param);
        }
        let distance = non_zero_sign(aq.cross(segment)) * endpoint_distance;
        let endpoint_alignment = segment
            .normalized()
            .dot(endpoint.normalized())
            .abs()
            .clamp(0.0, 1.0);
        (SignedDistance::new(distance, endpoint_alignment), param)
    }
}

#[derive(Clone, Copy, Debug)]
struct SignedDistance {
    distance: f32,
    dot: f32,
}

impl SignedDistance {
    fn new(distance: f32, dot: f32) -> Self {
        Self { distance, dot }
    }

    fn inf() -> Self {
        Self {
            distance: f32::INFINITY,
            dot: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ChannelSelector {
    min_distance: SignedDistance,
    near_segment_index: Option<usize>,
    near_param: f32,
}

impl Default for ChannelSelector {
    fn default() -> Self {
        Self {
            min_distance: SignedDistance::inf(),
            near_segment_index: None,
            near_param: 0.0,
        }
    }
}

fn distance_is_better(candidate: SignedDistance, current: SignedDistance) -> bool {
    if !candidate.distance.is_finite() {
        return false;
    }
    if !current.distance.is_finite() {
        return true;
    }
    let candidate_abs = candidate.distance.abs();
    let current_abs = current.distance.abs();
    if candidate_abs < current_abs {
        true
    } else if candidate_abs > current_abs {
        false
    } else {
        candidate.dot < current.dot
    }
}

fn non_zero_sign(value: f32) -> f32 {
    if value < 0.0 {
        -1.0
    } else {
        1.0
    }
}

fn pseudo_distance_from_cross(point_vec: Vec2, segment_vec: Vec2) -> f32 {
    let segment_len = segment_vec.length();
    if segment_len <= 0.000_000_1 {
        f32::INFINITY
    } else {
        point_vec.cross(segment_vec).abs() / segment_len
    }
}

fn signed_pseudo_distance(point_vec: Vec2, segment_vec: Vec2) -> f32 {
    non_zero_sign(point_vec.cross(segment_vec)) * pseudo_distance_from_cross(point_vec, segment_vec)
}

impl FlatSegment {
    fn signed_pseudo_distance_at_start(self, point: Point<f32>) -> f32 {
        let aq = Vec2::from_points(self.p0, point);
        let segment = Vec2::from_points(self.p0, self.p1);
        signed_pseudo_distance(aq, segment)
    }

    fn signed_pseudo_distance_at_end(self, point: Point<f32>) -> f32 {
        let bq = Vec2::from_points(self.p1, point);
        let segment = Vec2::from_points(self.p0, self.p1);
        signed_pseudo_distance(bq, segment)
    }
}

impl FlatSegment {
    fn distance_to_pseudo_distance(
        self,
        distance: &mut SignedDistance,
        point: Point<f32>,
        param: f32,
    ) {
        let segment = Vec2::from_points(self.p0, self.p1);
        let segment_len_sq = segment.dot(segment);
        if segment_len_sq <= 0.000_000_1 {
            return;
        }

        if param < 0.0 {
            let aq = Vec2::from_points(self.p0, point);
            let ts = aq.dot(segment) / segment_len_sq;
            if ts < 0.0 {
                let pseudo_distance = self.signed_pseudo_distance_at_start(point);
                if pseudo_distance.abs() <= distance.distance.abs() {
                    distance.distance = pseudo_distance;
                    distance.dot = 0.0;
                }
            }
        } else if param > 1.0 {
            let bq = Vec2::from_points(self.p1, point);
            let ts = bq.dot(segment) / segment_len_sq;
            if ts > 0.0 {
                let pseudo_distance = self.signed_pseudo_distance_at_end(point);
                if pseudo_distance.abs() <= distance.distance.abs() {
                    distance.distance = pseudo_distance;
                    distance.dot = 0.0;
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Vec2 {
    x: f32,
    y: f32,
}

impl Vec2 {
    fn from_points(p0: Point<f32>, p1: Point<f32>) -> Self {
        Self {
            x: p1.x - p0.x,
            y: p1.y - p0.y,
        }
    }

    fn is_zero(self) -> bool {
        const EPS: f32 = 0.000_001;
        self.x.abs() <= EPS && self.y.abs() <= EPS
    }

    fn normalized(self) -> Self {
        let length_sq = self.x * self.x + self.y * self.y;
        if length_sq <= 0.000_000_1 {
            return Self::default();
        }
        let inv_length = length_sq.sqrt().recip();
        Self {
            x: self.x * inv_length,
            y: self.y * inv_length,
        }
    }

    fn orthonormal(self, polarity: bool) -> Self {
        let length = self.length();
        if length <= 0.000_000_1 {
            return Self::default();
        }
        let inv_length = length.recip();
        if polarity {
            Self {
                x: -self.y * inv_length,
                y: self.x * inv_length,
            }
        } else {
            Self {
                x: self.y * inv_length,
                y: -self.x * inv_length,
            }
        }
    }

    fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    fn cross(self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }

    fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use {
        super::*,
        crate::text::{
            geom::Rect,
            glyph_outline,
            image::Image,
        },
        rustybuzz::ttf_parser,
        std::time::{Duration, Instant},
    };

    fn outlines(font_file: &str, text: &str) -> Vec<(char, GlyphOutline)> {
        let path = format!("{}/../widgets/resources/{font_file}", env!("CARGO_MANIFEST_DIR"));
        let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{path}: {error}"));
        let face = ttf_parser::Face::parse(&bytes, 0).unwrap();
        let units_per_em = face.units_per_em() as f32;
        text.chars()
            .filter_map(|ch| {
                let id = face.glyph_index(ch)?;
                let mut builder = glyph_outline::Builder::new();
                let bounds = face.outline_glyph(id, &mut builder)?;
                let min = Point::new(bounds.x_min as f32, bounds.y_min as f32);
                let max = Point::new(bounds.x_max as f32, bounds.y_max as f32);
                Some((ch, builder.finish(Rect::new(min, max - min), units_per_em)))
            })
            .collect()
    }

    fn render(outline: &GlyphOutline, dpxs_per_em: f32, cull: bool) -> (Vec<Bgra>, Duration) {
        // The layouter's MSDF settings, and the rasterizer's image size.
        let settings = Settings { padding: 4, radius: 8.0, cutoff: 0.25, corner_angle_threshold: 3.0 };
        let in_dpxs = outline.size_in_ems() * dpxs_per_em;
        let size = Size::new(
            in_dpxs.width.ceil() as usize + settings.padding * 2,
            in_dpxs.height.ceil() as usize + settings.padding * 2,
        );
        let mut image = Image::<Bgra>::new(size);
        let started = Instant::now();
        Msdfer::new(settings).render(outline, dpxs_per_em, &mut image.subimage_mut(Rect::from(size)), cull);
        (image.into_pixels(), started.elapsed())
    }

    /// Culling must not change one bit of any channel; it only skips segments
    /// that cannot be nearest. Run with `--nocapture` in release for timings.
    #[test]
    fn culled_msdf_matches_full_scan() {
        let glyphs = [
            ("IBMPlexSans-Text.ttf", "AaBgQW@&%?8Rß"),
            ("LXGWWenKaiRegular.ttf", "这样的周末真好下次一起测试输入是否流畅龍鬱齉體驚"),
        ];
        let (mut full_time, mut culled_time) = (Duration::ZERO, Duration::ZERO);
        let mut images = 0;
        for (font, text) in glyphs {
            for (ch, outline) in outlines(font, text) {
                for dpxs_per_em in [20.0, 32.0, 48.0, 64.0, 96.0, 128.0] {
                    let (full, full_elapsed) = render(&outline, dpxs_per_em, false);
                    let (culled, culled_elapsed) = render(&outline, dpxs_per_em, true);
                    assert!(full == culled, "{ch:?} at {dpxs_per_em} dpx/em differs");
                    full_time += full_elapsed;
                    culled_time += culled_elapsed;
                    images += 1;
                }
            }
        }
        assert!(images > 100);
        println!(
            "{images} glyph images: full scan {:.1} ms, culled {:.1} ms ({:.1}x)",
            full_time.as_secs_f64() * 1e3,
            culled_time.as_secs_f64() * 1e3,
            full_time.as_secs_f64() / culled_time.as_secs_f64(),
        );
    }
}
