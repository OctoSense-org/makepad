//! `"<xml>".parse_feed(style)`: RSS 2.0 or Atom into an array of rows,
//! `{title, link, source, published, summary, image}`, for an app that reads
//! feeds. Parsing in script would spend an app's instruction budget on string
//! scanning; this is pure computation, so every isolate may call it.
//!
//! `style` is `"google"` (titles end in ` - Source`), `"digest"` (TechMeme:
//! titles end in ` (Outlet)`) or anything else for plain feeds. A document
//! that is not a feed yields `nil`; a malformed one yields fewer rows.
//!
//! Items are split on `<item` / `<entry`, fields are the first matching
//! element (CDATA unwrapped, entities decoded, tags stripped, whitespace
//! collapsed), without an XML crate. Known limitation: an `<item` or
//! `<title` inside a CDATA description can confuse the splitter, which costs
//! that item its summary or title, never a panic.
use crate::heap::*;
use crate::makepad_live_id::live_id::*;
use crate::native::*;
use crate::value::*;
use crate::*;

const SUMMARY_CHARS: usize = 280;

fn is_http_url(s: &str) -> bool {
    ["http://", "https://"].iter().any(|scheme| s.get(..scheme.len()).is_some_and(|head| head.eq_ignore_ascii_case(scheme)))
}

/// One story, whatever the feed.
#[derive(Clone, Debug, Default, PartialEq)]
struct Headline {
    title: String,
    link: String,
    source: String,
    source_id: String,
    published: Option<i64>,
    points: Option<u32>,
    comments: Option<u32>,
    summary: String,
    discussion: Option<String>,
    image: Option<String>,
}

fn json_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn rows_json(rows: &[Headline]) -> String {
    let mut out = String::from("[");
    for (i, row) in rows.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str("{\"title\":");
        json_string(&mut out, &row.title);
        out.push_str(",\"link\":");
        json_string(&mut out, &row.link);
        out.push_str(",\"source\":");
        json_string(&mut out, &row.source);
        out.push_str(",\"summary\":");
        json_string(&mut out, &row.summary);
        out.push_str(",\"image\":");
        json_string(&mut out, row.image.as_deref().unwrap_or(""));
        out.push_str(",\"published\":");
        match row.published {
            Some(secs) => out.push_str(&secs.to_string()),
            None => out.push('0'),
        }
        out.push('}');
    }
    out.push(']');
    out
}

pub fn define_feed_module(heap: &mut ScriptHeap, native: &mut ScriptNative) {
    native.add_type_method(
        heap,
        ScriptValueType::REDUX_STRING,
        id!(parse_feed),
        &[(id!(style), NIL)],
        |vm, args| {
            let sself = script_value!(vm, args.self);
            let style_value = script_value!(vm, args.style);
            let style = vm.bx.heap.string_with(style_value, |_, s| s.to_string()).unwrap_or_default();
            let style = match style.as_str() {
                "google" => Style { split_source: true, digest: false },
                "digest" => Style { split_source: false, digest: true },
                _ => Style::default(),
            };
            let Some(parsed) = vm.bx.heap.string_with(sself, |_, s| parse_styled(s, style)) else {
                return script_err_unexpected!(vm.bx.threads.cur_ref().trap, "parse_feed called on non-string value");
            };
            let Ok(rows) = parsed else { return NIL };
            let json = rows_json(&rows);
            let mut json_parser = std::mem::take(&mut vm.bx.threads.cur().json_parser);
            let result = json_parser.read_json(&json, &mut vm.bx.heap);
            vm.bx.threads.cur().json_parser = json_parser;
            result
        },
    );
}



/// How a feed spells its stories, beyond plain RSS or Atom.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Style {
    /// Google News: titles end in ` - Source`.
    split_source: bool,
    /// A digest (TechMeme): titles end in ` (Author/Outlet)` and the
    /// description is `Outlet: headline` again.
    digest: bool,
}

fn parse_styled(xml: &str, style: Style) -> Result<Vec<Headline>, String> {
    // A feed that prefixes every Atom element (`<atom:entry>`) reads like
    // a bare one once the prefix is dropped.
    let unprefixed;
    let xml = if !xml.contains("<entry") && xml.contains("<atom:entry") {
        unprefixed = xml.replace("<atom:", "<").replace("</atom:", "</");
        unprefixed.as_str()
    } else {
        xml
    };
    let (open, close, is_atom) = feed_kind(xml).ok_or_else(|| "not an RSS or Atom feed".to_string())?;
    let rows = xml
        .split(open)
        .skip(1)
        .filter_map(|chunk| {
            // `chunk` starts right after `<item` / `<entry`: attributes or `>`.
            let body = &chunk[chunk.find('>')? + 1..];
            let body = body.split_once(close).map_or(body, |(b, _)| b);
            parse_item(body, is_atom, style)
        })
        .collect();
    Ok(rows)
}

const RSS: (&str, &str, bool) = ("<item", "</item>", false);
const ATOM: (&str, &str, bool) = ("<entry", "</entry>", true);

/// The item element to split on and whether the feed is Atom: the root
/// element decides (`<feed>`; `<rss>` or `<rdf:RDF>`), else the first item
/// element found.
fn feed_kind(xml: &str) -> Option<(&'static str, &'static str, bool)> {
    match root_element(xml) {
        Some("feed") => Some(ATOM),
        Some("rss" | "rdf:RDF") => Some(RSS),
        _ if has_element(xml, "item") => Some(RSS),
        _ if has_element(xml, "entry") => Some(ATOM),
        _ => None,
    }
}

/// The name of the root element: the first element after the prolog (a
/// byte-order mark, whitespace, `<?...?>`, `<!--...-->`, `<!DOCTYPE ...>`).
/// Looking only there keeps a `<feed>` inside a description from naming
/// the kind.
fn root_element(xml: &str) -> Option<&str> {
    let prolog_space = |c: char| c.is_whitespace() || c == '\u{feff}';
    let mut rest = xml.trim_start_matches(prolog_space);
    loop {
        let after_prolog = if let Some(after) = rest.strip_prefix("<?") {
            after.split_once("?>")?.1
        } else if let Some(after) = rest.strip_prefix("<!--") {
            after.split_once("-->")?.1
        } else if let Some(after) = rest.strip_prefix("<!") {
            // A DOCTYPE with an internal subset (`[...]`) ends at its first
            // inner `>`; what follows is not an element, so the caller falls
            // back to the item heuristic. Feeds do not carry one.
            after.split_once('>')?.1
        } else {
            break;
        };
        rest = after_prolog.trim_start_matches(prolog_space);
    }
    let name = rest.strip_prefix('<')?;
    let end = name.find(|c: char| c.is_whitespace() || c == '>' || c == '/').unwrap_or(name.len());
    Some(&name[..end]).filter(|n| !n.is_empty())
}

/// Whether `xml` opens an element named exactly `name`.
fn has_element(xml: &str, name: &str) -> bool {
    let open = format!("<{name}");
    let mut from = 0;
    while let Some(pos) = xml[from..].find(&open) {
        from += pos + open.len();
        if xml[from..].chars().next().is_some_and(|c| c == '>' || c == '/' || c.is_whitespace()) {
            return true;
        }
    }
    false
}

/// One `<item>` / `<entry>` body as a row; None without a title or an http
/// link.
fn parse_item(body: &str, is_atom: bool, style: Style) -> Option<Headline> {
    let mut title = clean_text(element_text(body, "title")?);
    if title.is_empty() {
        return None;
    }
    let link = if is_atom { atom_link(body) } else { rss_link(body) }?;
    let mut source = if is_atom {
        String::new()
    } else {
        element_text(body, "source").map(clean_text).unwrap_or_default()
    };
    if style.split_source {
        if let Some((head, tail)) = split_google_source(&title) {
            title = head;
            source = tail;
        }
    }
    if style.digest {
        if let Some((head, outlet)) = trailing_outlet(&title) {
            title = head;
            source = outlet;
        }
    }
    let mut summary = element_text(body, "description")
        .or_else(|| element_text(body, "summary"))
        .or_else(|| element_text(body, "content"))
        .or_else(|| element_text(body, "content:encoded"))
        .map(|d| truncate(&clean_text(d), SUMMARY_CHARS))
        .unwrap_or_default();
    // A description of `Outlet: headline` (a digest's): the outlet is the
    // row's publisher when nothing named one, and a description that is
    // only the headline again is no summary.
    if let Some((outlet, rest)) = outlet_prefix(&title, &summary) {
        if source.is_empty() {
            source = outlet;
        }
        summary = rest;
    }
    let published = element_text(body, "pubDate")
        .and_then(|d| parse_rfc822(d.trim()))
        .or_else(|| element_text(body, "published").and_then(|d| parse_rfc3339(d.trim())))
        .or_else(|| element_text(body, "updated").and_then(|d| parse_rfc3339(d.trim())))
        .or_else(|| element_text(body, "dc:date").and_then(|d| parse_rfc3339(d.trim())));
    // The model stamps `source_id` when the rows land: one reader serves
    // every feed.
    Some(Headline {
        title,
        link,
        source,
        source_id: String::new(),
        published,
        points: None,
        comments: None,
        summary,
        discussion: None,
        image: image_url(body),
    })
}

/// The story's picture, when the item carries one: a `<media:content>`
/// that is an image (its `medium` or `type` says so, or neither is given),
/// a `<media:thumbnail>`, an `<enclosure>` with an image type, else the
/// first `<img>` in the description or content. Only an http(s) URL
/// counts, and it is asked for over https whatever the feed says: feeds
/// still carry plain http picture links (TechMeme's), the app's own
/// connections are held to transport security on Apple platforms, and a
/// picture host without https costs the story its picture, not its row.
fn image_url(body: &str) -> Option<String> {
    media_image(body, "media:content", true)
        .or_else(|| media_image(body, "media:thumbnail", false))
        .or_else(|| media_image(body, "enclosure", true))
        .or_else(|| inline_image(body))
        .map(|url| match url.strip_prefix("http://") {
            Some(rest) => format!("https://{rest}"),
            None => url,
        })
}

/// The `url` of the first `<tag>` element that is an image. `typed`: the
/// element carries a `medium` or `type`, and it must say image when it
/// does (an enclosure must say so; a media:content with neither is taken
/// as one). Every such element is looked at, not just the first: a feed
/// may list a video before its still.
fn media_image(body: &str, tag: &str, typed: bool) -> Option<String> {
    let open = format!("<{tag}");
    let mut from = 0;
    while let Some(pos) = body[from..].find(&open) {
        let at = from + pos;
        let after = &body[at + open.len()..];
        let next = after.chars().next()?;
        from = at + open.len();
        if next != '>' && next != '/' && !next.is_whitespace() {
            continue;
        }
        let tag_end = after.find('>')?;
        let attrs = &after[..tag_end];
        if typed {
            let medium = attribute(attrs, "medium");
            let kind = attribute(attrs, "type");
            let says_image = medium.as_deref() == Some("image") || kind.as_deref().is_some_and(|t| t.starts_with("image/"));
            let says_nothing = medium.is_none() && kind.is_none() && tag == "media:content";
            if !says_image && !says_nothing {
                continue;
            }
        }
        if let Some(url) = attribute(attrs, "url").map(|u| decode_entities(&u)) {
            if is_http_url(&url) {
                return Some(url);
            }
        }
    }
    None
}

/// The first `<img src>` in the item's HTML description or content that is
/// not a tiny one (a permalink icon, a tracking pixel: anything whose
/// `width` or `height` says under `MIN_IMAGE_PX`). Tags and attributes in
/// any case, values quoted or not.
fn inline_image(body: &str) -> Option<String> {
    let html = element_text(body, "description")
        .or_else(|| element_text(body, "content:encoded"))
        .or_else(|| element_text(body, "content"))?;
    // CDATA markers dropped and entities decoded: feeds double-encode HTML.
    let html = decode_entities(&html.replace("<![CDATA[", "").replace("]]>", ""));
    // ASCII lowercasing keeps every byte offset, so positions found in the
    // lowered copy index the original.
    let lower = html.to_ascii_lowercase();
    let mut from = 0;
    while let Some(pos) = lower[from..].find("<img") {
        let at = from + pos + 4;
        let Some(tag_end) = lower[at..].find('>') else { break };
        let attrs = &html[at..at + tag_end];
        from = at + tag_end;
        if is_tiny(attrs) {
            continue;
        }
        if let Some(src) = attribute_ci(attrs, "src").map(|u| decode_entities(&u)) {
            if is_http_url(&src) {
                return Some(src);
            }
        }
    }
    None
}

/// Under this many pixels on a side an image is an icon, not a picture.
const MIN_IMAGE_PX: f64 = 50.0;

/// Whether an `<img>`'s `width` or `height` attribute says it is an icon.
fn is_tiny(attrs: &str) -> bool {
    ["width", "height"].iter().any(|key| {
        attribute_ci(attrs, key)
            .and_then(|v| v.trim().trim_end_matches("px").trim().parse::<f64>().ok())
            .is_some_and(|px| px < MIN_IMAGE_PX)
    })
}

/// `attribute` for HTML: the name in any case, the value quoted or bare
/// (ending at whitespace).
fn attribute_ci(attrs: &str, name: &str) -> Option<String> {
    let lower = attrs.to_ascii_lowercase();
    let key = format!("{}=", name.to_ascii_lowercase());
    let mut from = 0;
    while let Some(pos) = lower[from..].find(&key) {
        let at = from + pos;
        let before_ok = at == 0 || lower[..at].ends_with(|c: char| c.is_whitespace());
        from = at + key.len();
        if !before_ok {
            continue;
        }
        let rest = &attrs[at + key.len()..];
        let value = match rest.chars().next()? {
            quote @ ('"' | '\'') => {
                let value = &rest[1..];
                &value[..value.find(quote)?]
            }
            _ => rest.split(|c: char| c.is_whitespace()).next().unwrap_or(""),
        };
        return Some(value.to_string());
    }
    None
}

/// `Outlet: headline …` in a description whose remainder opens with the
/// item's own title: the outlet, and what is left after the headline
/// (empty when the description was the headline alone). None when the
/// description does not have that shape.
fn outlet_prefix(title: &str, summary: &str) -> Option<(String, String)> {
    // The colon alone: a stripped `<BR>` may have left no space after it.
    let (outlet, rest) = summary.split_once(':')?;
    let outlet = outlet.trim();
    let rest = rest.trim_start();
    if !looks_like_name(outlet) {
        return None;
    }
    let title = title.trim().trim_end_matches('…').trim();
    let head: String = title.chars().take(TITLE_MATCH_CHARS).collect();
    if head.is_empty() || !rest.to_lowercase().starts_with(&head.to_lowercase()) {
        return None;
    }
    let rest = rest.trim();
    // What follows the headline, compared character by character (a
    // lowercase form can differ in length from its original, so byte
    // offsets of one never index the other).
    let remainder = strip_prefix_ci(rest, title).unwrap_or("").trim();
    let remainder = remainder.trim_start_matches(['—', '-', '·', ':', ',']).trim();
    Some((outlet.to_string(), remainder.to_string()))
}

/// `text` after `prefix`, matched case-insensitively character by
/// character; None when `text` does not start with it.
fn strip_prefix_ci<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    let mut rest = text;
    for want in prefix.chars() {
        let got = rest.chars().next()?;
        if !got.to_lowercase().eq(want.to_lowercase()) {
            return None;
        }
        rest = &rest[got.len_utf8()..];
    }
    Some(rest)
}

/// A digest title's trailing ` (Author/Outlet)` or ` (Outlet)`: the title
/// without it, and the outlet (the last `/`-separated part). None when the
/// title does not end that way, or what is in the parentheses is too long
/// to be a name.
fn trailing_outlet(title: &str) -> Option<(String, String)> {
    let title = title.trim();
    let inner_end = title.strip_suffix(')')?;
    let open = inner_end.rfind('(')?;
    let inner = inner_end[open + 1..].trim();
    let head = inner_end[..open].trim_end();
    if inner.is_empty() || head.is_empty() || inner.chars().count() > OUTLET_CHARS || inner.contains('(') {
        return None;
    }
    let outlet = inner.rsplit('/').next().unwrap_or(inner).trim();
    if outlet.is_empty() {
        return None;
    }
    Some((head.to_string(), outlet.to_string()))
}

/// Whether `text` reads as an outlet's name rather than the start of a
/// sentence: a few words, each capitalised or a number (`Financial Times`,
/// `9to5Mac`, `The Verge`), connectors aside; no sentence punctuation.
fn looks_like_name(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.is_empty() || text.chars().count() > OUTLET_CHARS || text.contains(['.', '!', '?']) {
        return false;
    }
    words.iter().all(|word| {
        matches!(*word, "of" | "the" | "and" | "de" | "for" | "&")
            || word.chars().next().is_some_and(|c| c.is_uppercase() || c.is_ascii_digit())
    })
}

/// An outlet's name is a few words: anything longer is a sentence with a
/// colon in it.
const OUTLET_CHARS: usize = 40;
/// How much of the headline the description must open with.
const TITLE_MATCH_CHARS: usize = 24;

/// Google News titles end in ` - Source`; the last dash is the separator.
fn split_google_source(title: &str) -> Option<(String, String)> {
    let (head, tail) = title.rsplit_once(" - ")?;
    let (head, tail) = (head.trim(), tail.trim());
    (!head.is_empty() && !tail.is_empty()).then(|| (head.to_string(), tail.to_string()))
}

/// The first `<tag ...>...</tag>` in `xml` as its attribute text and raw
/// inner text. A tag name must end at `>`, whitespace or `/`, so `title`
/// never matches `titleX`. A self-closing element has empty inner text.
fn element<'a>(xml: &'a str, tag: &str) -> Option<(&'a str, &'a str)> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let mut from = 0;
    while let Some(pos) = xml[from..].find(&open) {
        let at = from + pos;
        let after = &xml[at + open.len()..];
        let next = after.chars().next()?;
        if next != '>' && next != '/' && !next.is_whitespace() {
            from = at + 1;
            continue;
        }
        let tag_end = after.find('>')?;
        let attrs = after[..tag_end].trim_end();
        if let Some(attrs) = attrs.strip_suffix('/') {
            return Some((attrs, ""));
        }
        let content = &after[tag_end + 1..];
        let end = content.find(&close)?;
        return Some((attrs, &content[..end]));
    }
    None
}

/// The raw inner text of the first `<tag>` in `xml`.
fn element_text<'a>(xml: &'a str, tag: &str) -> Option<&'a str> {
    element(xml, tag).map(|(_, text)| text)
}

/// An RSS item's `<link>` when it is an http(s) URL, else its `<guid>` when
/// that is one and not marked `isPermaLink="false"`.
fn rss_link(body: &str) -> Option<String> {
    element_text(body, "link")
        .map(clean_text)
        .filter(|l| is_http_url(l))
        .or_else(|| permalink_guid(body))
}

fn permalink_guid(body: &str) -> Option<String> {
    let (attrs, text) = element(body, "guid")?;
    if attribute(attrs, "isPermaLink").is_some_and(|v| v.eq_ignore_ascii_case("false")) {
        return None;
    }
    Some(clean_text(text)).filter(|g| is_http_url(g))
}

/// Atom links are `<link href="..." rel="..."/>`; the article is the one
/// without `rel` or with `rel="alternate"`.
fn atom_link(xml: &str) -> Option<String> {
    let mut from = 0;
    while let Some(pos) = xml[from..].find("<link") {
        let at = from + pos;
        let after = &xml[at + 5..];
        let tag_end = after.find('>')?;
        let attrs = &after[..tag_end];
        // Past the whole tag: restarting inside it would rescan to its `>`
        // once per `<link` in a wall of them.
        from = at + 5 + tag_end + 1;
        let rel = attribute(attrs, "rel");
        if rel.as_deref().is_some_and(|r| r != "alternate") {
            continue;
        }
        if let Some(href) = attribute(attrs, "href") {
            let href = decode_entities(&href);
            if is_http_url(&href) {
                return Some(href);
            }
        }
    }
    None
}

fn attribute(attrs: &str, name: &str) -> Option<String> {
    let key = format!("{name}=");
    let mut from = 0;
    while let Some(pos) = attrs[from..].find(&key) {
        let at = from + pos;
        let before_ok = at == 0 || attrs[..at].ends_with(|c: char| c.is_whitespace());
        let rest = &attrs[at + key.len()..];
        from = at + key.len();
        if !before_ok {
            continue;
        }
        let quote = rest.chars().next()?;
        if quote != '"' && quote != '\'' {
            continue;
        }
        let value = &rest[1..];
        let end = value.find(quote)?;
        return Some(value[..end].to_string());
    }
    None
}

/// CDATA markers dropped (they can sit anywhere in a description), entities
/// decoded, HTML tags stripped, entities decoded again (feeds double-encode
/// HTML in descriptions), whitespace collapsed.
pub(crate) fn clean_text(raw: &str) -> String {
    let text = raw.replace("<![CDATA[", "").replace("]]>", "");
    let text = decode_entities(text.trim());
    let text = strip_tags(&text);
    let text = decode_entities(&text);
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// What an angle-bracketed run turns into once stripped.
enum TagKind {
    /// Wraps words inside a sentence: dropped, so `<a>Example</a>:` keeps
    /// its colon attached.
    Inline,
    /// Separates words or lines: a space, so `line<br/>break` never fuses.
    Block,
    /// Not an HTML tag at all: a title may well say `<3` or `x < y`.
    Text,
}

const INLINE_TAGS: &[&str] = &[
    "a", "abbr", "b", "big", "cite", "code", "del", "em", "font", "i", "ins", "mark", "q", "s", "small", "span",
    "strike", "strong", "sub", "sup", "tt", "u",
];

const BLOCK_TAGS: &[&str] = &[
    "article", "aside", "audio", "blockquote", "br", "center", "dd", "div", "dl", "dt", "figcaption", "figure",
    "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "iframe", "img", "li", "nav", "ol", "p", "picture",
    "pre", "section", "source", "table", "tbody", "td", "th", "thead", "tr", "ul", "video",
];

fn tag_kind(tag: &str) -> TagKind {
    // Comments and processing instructions are markup too.
    if tag.starts_with('!') || tag.starts_with('?') {
        return TagKind::Block;
    }
    let name = tag
        .trim_start_matches('/')
        .split(|c: char| c.is_whitespace() || c == '/')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    if INLINE_TAGS.contains(&name.as_str()) {
        TagKind::Inline
    } else if BLOCK_TAGS.contains(&name.as_str()) {
        TagKind::Block
    } else {
        TagKind::Text
    }
}

/// Remove known HTML tags (see `TagKind`); an unclosed `<` is text.
fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find('<') {
        out.push_str(&rest[..open]);
        let after = &rest[open..];
        let Some(close) = after.find('>') else {
            rest = after;
            break;
        };
        match tag_kind(&after[1..close]) {
            TagKind::Inline => {}
            TagKind::Block => out.push(' '),
            TagKind::Text => out.push_str(&after[..=close]),
        }
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    out
}

/// The scan for an entity's `;` looks at `&` and the ten bytes after it,
/// so an entity is at most nine characters between `&` and `;`; the
/// longest wanted, `&#x10FFFF;`, has eight.
const MAX_ENTITY_LEN: usize = 10;

/// The named entities feeds actually use, plus numeric ones. A numeric
/// entity for a control character (other than tab, newline and carriage
/// return) stays as written, so no feed can smuggle one into a label.
pub(crate) fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find('&') {
        out.push_str(&rest[..pos]);
        let after = &rest[pos..];
        // Only look as far as an entity can reach, so a wall of `&` stays linear.
        let Some(end) = after.bytes().take(MAX_ENTITY_LEN + 1).position(|b| b == b';') else {
            out.push('&');
            rest = &after[1..];
            continue;
        };
        let entity = &after[1..end];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some(' '),
            "mdash" => Some('—'),
            "ndash" => Some('–'),
            "hellip" => Some('…'),
            "lsquo" => Some('‘'),
            "rsquo" => Some('’'),
            "ldquo" => Some('“'),
            "rdquo" => Some('”'),
            _ => numeric_entity(entity),
        };
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &after[end + 1..];
            }
            None => {
                out.push('&');
                rest = &after[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// `#65` or `#x41`, digits only (no sign), and never a control character
/// other than tab, newline or carriage return.
fn numeric_entity(entity: &str) -> Option<char> {
    let number = entity.strip_prefix('#')?;
    let code = match number.strip_prefix(['x', 'X']) {
        Some(hex) if !hex.is_empty() && hex.bytes().all(|b| b.is_ascii_hexdigit()) => u32::from_str_radix(hex, 16).ok()?,
        Some(_) => return None,
        None if !number.is_empty() && number.bytes().all(|b| b.is_ascii_digit()) => number.parse().ok()?,
        None => return None,
    };
    char::from_u32(code).filter(|c| !c.is_control() || matches!(c, '\n' | '\t' | '\r'))
}

/// At most `max` characters: cut to `max - 1`, a trailing space dropped, an
/// ellipsis appended.
pub(crate) fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let cut: String = text.chars().take(max.saturating_sub(1)).collect();
    format!("{}…", cut.trim_end())
}

const MONTHS: [&str; 12] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"];

/// `Wed, 16 Sep 2026 08:15:01 -0400`, the weekday and seconds optional,
/// the zone `GMT`/`UT`/`UTC`/`Z`, `±HHMM`, or a US abbreviation.
pub(crate) fn parse_rfc822(text: &str) -> Option<i64> {
    let mut parts: Vec<&str> = text.split_whitespace().collect();
    if parts.first().is_some_and(|p| p.ends_with(',')) {
        parts.remove(0);
    }
    if parts.len() < 4 {
        return None;
    }
    let day: i64 = parts[0].parse().ok()?;
    let month_name = parts[1].to_lowercase();
    let month = MONTHS.iter().position(|m| month_name.starts_with(m))? as i64 + 1;
    let year: i64 = parts[2].parse().ok()?;
    let year = if year < 100 { year + 2000 } else { year };
    let clock = hms(parts[3])?;
    let offset = parts.get(4).map(|z| zone_offset(z)).unwrap_or(0);
    unix_seconds(year, month, day, clock, offset)
}

/// `2026-09-16T08:30:00+02:00`, `...Z`, fractional seconds allowed, a
/// missing zone read as UTC.
pub(crate) fn parse_rfc3339(text: &str) -> Option<i64> {
    let (date, time) = text.split_once(['T', 't', ' '])?;
    let mut d = date.split('-');
    let year: i64 = d.next()?.parse().ok()?;
    let month: i64 = d.next()?.parse().ok()?;
    let day: i64 = d.next()?.parse().ok()?;
    let zone_at = time.find(['Z', 'z', '+']).or_else(|| time.rfind('-')).unwrap_or(time.len());
    let (clock, zone) = time.split_at(zone_at);
    let clock = hms(clock.split('.').next()?)?;
    unix_seconds(year, month, day, clock, zone_offset(zone))
}

/// Unix seconds for a civil date and clock in a zone `offset` seconds east
/// of UTC; None when a date field is out of range. A day past its month's
/// end is let through, as a feed's own clock would have it.
fn unix_seconds(year: i64, month: i64, day: i64, (h, m, s): (i64, i64, i64), offset: i64) -> Option<i64> {
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }
    Some(days_from_civil(year, month, day) * 86_400 + h * 3600 + m * 60 + s - offset)
}

/// `HH:MM` or `HH:MM:SS`, each field in range (a leap second allowed).
fn hms(text: &str) -> Option<(i64, i64, i64)> {
    let mut p = text.split(':');
    let h: i64 = p.next()?.parse().ok()?;
    let m: i64 = p.next()?.parse().ok()?;
    let s: i64 = match p.next() {
        Some(s) => s.parse().ok()?,
        None => 0,
    };
    ((0..=23).contains(&h) && (0..=59).contains(&m) && (0..=60).contains(&s)).then_some((h, m, s))
}

/// Seconds east of UTC for a zone token. Zone names other than the UTC
/// spellings and the US abbreviations (CET, BST, AEST, ...) are read as
/// UTC: a time off by a few hours beats no time at all.
fn zone_offset(zone: &str) -> i64 {
    match zone.to_uppercase().as_str() {
        "GMT" | "UT" | "UTC" | "Z" | "" => 0,
        "EST" => -5 * 3600,
        "EDT" => -4 * 3600,
        "CST" => -6 * 3600,
        "CDT" => -5 * 3600,
        "MST" => -7 * 3600,
        "MDT" => -6 * 3600,
        "PST" => -8 * 3600,
        "PDT" => -7 * 3600,
        z => {
            let sign = if z.starts_with('-') { -1 } else { 1 };
            let digits: String = z.chars().filter(|c| c.is_ascii_digit()).collect();
            if digits.len() < 4 {
                return 0;
            }
            let hh: i64 = digits[..2].parse().unwrap_or(0);
            let mm: i64 = digits[2..4].parse().unwrap_or(0);
            sign * (hh * 3600 + mm * 60)
        }
    }
}

/// Days since 1970-01-01 for a proleptic Gregorian date.
pub(crate) fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}


#[cfg(test)]
mod tests {
    use super::*;

    const RSS: &str = r#"<?xml version="1.0"?><rss><channel><title>T</title>
      <item><title><![CDATA[Rust 2.0 &amp; you]]></title><link>https://example.com/a</link>
        <description>&lt;p&gt;Big &lt;b&gt;news&lt;/b&gt;&lt;/p&gt;</description>
        <pubDate>Tue, 10 Jun 2025 12:00:00 GMT</pubDate>
        <media:thumbnail url="http://img.example.com/a.jpg"/></item>
      <item><title>Second - The Outlet</title><link>https://example.com/b</link></item>
      <item><title></title><link>https://example.com/c</link></item>
    </channel></rss>"#;

    const ATOM: &str = r#"<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Atom post</title>
      <link rel="alternate" href="https://example.org/p"/><updated>2025-06-10T12:00:00Z</updated>
      <summary>Short</summary></entry></feed>"#;

    #[test]
    fn rss_items_become_rows_with_cleaned_text_times_and_https_images() {
        let rows = parse_styled(RSS, Style::default()).unwrap();
        assert_eq!(rows.len(), 2, "an untitled item is skipped");
        assert_eq!(rows[0].title, "Rust 2.0 & you");
        assert_eq!(rows[0].summary, "Big news");
        assert_eq!(rows[0].published, Some(1_749_556_800));
        assert_eq!(rows[0].image.as_deref(), Some("https://img.example.com/a.jpg"));
        assert_eq!(rows[1].title, "Second - The Outlet", "plain style keeps the title whole");
    }

    #[test]
    fn google_style_takes_the_outlet_off_the_title() {
        let rows = parse_styled(RSS, Style { split_source: true, digest: false }).unwrap();
        assert_eq!(rows[1].title, "Second");
        assert_eq!(rows[1].source, "The Outlet");
    }

    #[test]
    fn atom_entries_parse_and_non_feeds_are_refused() {
        let rows = parse_styled(ATOM, Style::default()).unwrap();
        assert_eq!(rows[0].link, "https://example.org/p");
        assert_eq!(rows[0].published, Some(1_749_556_800));
        assert!(parse_styled("<html><body>nope</body></html>", Style::default()).is_err());
    }

    #[test]
    fn rows_serialize_to_json_the_script_reader_accepts() {
        let rows = parse_styled(RSS, Style::default()).unwrap();
        let json = rows_json(&rows);
        assert!(json.starts_with("[{\"title\":\"Rust 2.0 & you\""), "{json}");
        let quoted = rows_json(&[Headline { title: "a \"q\"\n".into(), ..Default::default() }]);
        assert!(quoted.contains("\"a \\\"q\\\"\\n\""), "{quoted}");
    }
}
