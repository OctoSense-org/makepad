//! OSC 8 hyperlink storage.
//!
//! Cells carry a small index (`Cell::hyperlink`, 0 = none) into this table
//! instead of the URI, so a link printed across a thousand cells costs one
//! entry. Two links with the same explicit `id=` and URI share an entry
//! (the spec's way to tie separately printed pieces of one link together);
//! a link without an `id=` gets its own entry per `OSC 8` open, as in
//! Ghostty.
//!
//! The table is bounded and collected: rather than counting references on
//! every cell write, the terminal marks the ids still stored in any row
//! (scrollback included) and [`HyperlinkTable::retain`] frees the rest. A
//! collection runs when the table has doubled since the last one, so memory
//! stays within about twice the live links and the scan is amortised over
//! many opens; links on rows that scrolled out of the scrollback are freed
//! by the next collection. Reflow moves cells whole, ids included, so links
//! survive a resize untouched.

use std::collections::HashMap;

/// The longest URI accepted from `OSC 8` (bytes). Longer ones are refused.
pub const MAX_URI_LEN: usize = 2048;
/// The longest `id=` value accepted (bytes).
pub const MAX_ID_LEN: usize = 256;
/// The most links stored at once. When a collection cannot get below this,
/// new links print as plain text.
pub const MAX_LINKS: usize = 1 << 16;
/// The most URI bytes stored at once (the same rule applies).
pub const MAX_BYTES: usize = 8 << 20;
/// A collection runs no earlier than at this many links.
const MIN_GC_AT: usize = 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hyperlink {
    pub uri: Box<str>,
    /// The explicit `id=` the program gave, if any.
    pub id: Option<Box<str>>,
}

#[derive(Debug)]
pub struct HyperlinkTable {
    /// Entry `n - 1` is link id `n`.
    slots: Vec<Option<Hyperlink>>,
    free: Vec<u32>,
    /// Explicit (id, uri) -> link id, so pieces of one link share an entry.
    by_key: HashMap<(Box<str>, Box<str>), u32>,
    live: usize,
    bytes: usize,
    gc_at: usize,
}

impl Default for HyperlinkTable {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            by_key: HashMap::new(),
            live: 0,
            bytes: 0,
            gc_at: MIN_GC_AT,
        }
    }
}

impl HyperlinkTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// The link a cell's id points at.
    pub fn get(&self, id: u32) -> Option<&Hyperlink> {
        let index = (id as usize).checked_sub(1)?;
        self.slots.get(index)?.as_ref()
    }

    /// Links currently stored.
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Size of the id space (the highest id ever handed out and not yet
    /// shrunk away), for sizing a mark set.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Whether the terminal should collect before the next [`intern`].
    ///
    /// [`intern`]: HyperlinkTable::intern
    pub fn needs_gc(&self) -> bool {
        self.live >= self.gc_at || self.live >= MAX_LINKS || self.bytes >= MAX_BYTES
    }

    /// The id for a link, reusing the entry of an identical explicit-id
    /// link. `None` when the table is full (the text then prints unlinked)
    /// or the link is invalid (see [`valid_uri`]).
    pub fn intern(&mut self, id: Option<&str>, uri: &str) -> Option<u32> {
        if !valid_uri(uri) || id.is_some_and(|id| id.len() > MAX_ID_LEN) {
            return None;
        }
        if let Some(id) = id {
            if let Some(&found) = self.by_key.get(&(Box::from(id), Box::from(uri))) {
                return Some(found);
            }
        }
        let cost = uri.len() + id.map_or(0, str::len);
        if self.live >= MAX_LINKS || self.bytes + cost > MAX_BYTES {
            return None;
        }
        let link = Hyperlink {
            uri: uri.into(),
            id: id.map(Box::from),
        };
        let slot = match self.free.pop() {
            Some(slot) => {
                self.slots[slot as usize - 1] = Some(link);
                slot
            }
            None => {
                self.slots.push(Some(link));
                self.slots.len() as u32
            }
        };
        if let Some(id) = id {
            self.by_key.insert((id.into(), uri.into()), slot);
        }
        self.live += 1;
        self.bytes += cost;
        Some(slot)
    }

    /// Free every link whose id is not marked in `used` (indexed by id; an
    /// id beyond its end counts as unused), then schedule the next
    /// collection for when the table has doubled.
    pub fn retain(&mut self, used: &[bool]) {
        for index in 0..self.slots.len() {
            let id = index as u32 + 1;
            if self.slots[index].is_none() || used.get(id as usize).copied().unwrap_or(false) {
                continue;
            }
            let link = self.slots[index].take().unwrap();
            self.live -= 1;
            self.bytes -= link.uri.len() + link.id.as_ref().map_or(0, |id| id.len());
            if let Some(key_id) = link.id {
                self.by_key.remove(&(key_id, link.uri));
            }
        }
        // Give back the tail of empty slots so ids stay small, then rebuild
        // the free list (lowest ids first, reused before new ones).
        while self.slots.last().is_some_and(Option::is_none) {
            self.slots.pop();
        }
        self.free = (1..=self.slots.len() as u32)
            .rev()
            .filter(|&id| self.slots[id as usize - 1].is_none())
            .collect();
        self.gc_at = (self.live * 2).max(MIN_GC_AT);
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }
}

/// A URI the table stores: non-empty, at most [`MAX_URI_LEN`] bytes, and
/// free of control characters (C0, DEL and C1), which could otherwise reach
/// a status line or an opener.
pub fn valid_uri(uri: &str) -> bool {
    !uri.is_empty() && uri.len() <= MAX_URI_LEN && !uri.chars().any(|c| c.is_control())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_ids_share_an_entry() {
        let mut t = HyperlinkTable::new();
        let a = t.intern(Some("x"), "https://a").unwrap();
        let b = t.intern(Some("x"), "https://a").unwrap();
        assert_eq!(a, b);
        // Same id, other URI: another link.
        let c = t.intern(Some("x"), "https://b").unwrap();
        assert_ne!(a, c);
        // No id: a new entry per open.
        let d = t.intern(None, "https://a").unwrap();
        let e = t.intern(None, "https://a").unwrap();
        assert_ne!(d, e);
        assert_eq!(t.len(), 4);
        assert_eq!(&*t.get(a).unwrap().uri, "https://a");
        assert_eq!(t.get(0), None);
    }

    #[test]
    fn retain_frees_unmarked_and_reuses_ids() {
        let mut t = HyperlinkTable::new();
        let ids: Vec<u32> = (0..5)
            .map(|i| t.intern(None, &format!("https://{i}")).unwrap())
            .collect();
        let keyed = t.intern(Some("k"), "https://k").unwrap();
        let mut used = vec![false; t.capacity() + 1];
        used[ids[1] as usize] = true;
        t.retain(&used);
        assert_eq!(t.len(), 1);
        assert!(t.get(ids[0]).is_none());
        assert!(t.get(keyed).is_none());
        assert_eq!(&*t.get(ids[1]).unwrap().uri, "https://1");
        // The trailing empty slots were given back; the lowest free id is
        // reused first.
        assert_eq!(t.capacity(), 2);
        assert_eq!(t.intern(None, "https://new"), Some(ids[0]));
        // A freed explicit key no longer resolves to its old id.
        let again = t.intern(Some("k"), "https://k").unwrap();
        assert_eq!(&*t.get(again).unwrap().uri, "https://k");
    }

    #[test]
    fn refuses_invalid_and_overlong() {
        let mut t = HyperlinkTable::new();
        assert_eq!(t.intern(None, ""), None);
        assert_eq!(t.intern(None, "https://a\u{7}b"), None);
        assert_eq!(t.intern(None, "https://a\u{9b}b"), None);
        assert_eq!(
            t.intern(None, &format!("https://{}", "a".repeat(MAX_URI_LEN))),
            None
        );
        assert!(t
            .intern(None, &format!("https://{}", "a".repeat(MAX_URI_LEN - 8)))
            .is_some());
        assert_eq!(
            t.intern(Some(&"i".repeat(MAX_ID_LEN + 1)), "https://a"),
            None
        );
    }

    #[test]
    fn bounded_by_count() {
        let mut t = HyperlinkTable::new();
        for i in 0..MAX_LINKS {
            assert!(t.intern(None, &format!("h:{i}")).is_some());
        }
        assert!(t.needs_gc());
        assert_eq!(t.intern(None, "h:over"), None);
        t.retain(&[]);
        assert!(t.is_empty());
        assert!(t.intern(None, "h:after").is_some());
    }
}
