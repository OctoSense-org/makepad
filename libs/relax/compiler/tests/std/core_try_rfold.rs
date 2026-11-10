// DoubleEndedIterator::try_rfold (stable since 1.27): try_fold from the back, stopping at the
// first short-circuit and leaving the rest of the iterator in place. The std here had rfold and
// rfind only. fallible-iterator 0.3's `Convert` and `IntoFallible` (src/lib.rs,
// `self.0.try_rfold(init, |acc, v| f(acc, v?))`) and itertools 0.15's `ProcessResults::rfold`
// (src/process_results_impl.rs) call it.
use std::convert::Infallible;
use std::ops::ControlFlow;

// fallible-iterator's shape: an iterator of Results folded from the back, `?` on each item
struct Convert<I>(I);
impl<T, E, I: DoubleEndedIterator<Item = Result<T, E>>> Convert<I> {
    fn try_rfold<B, E2, F>(&mut self, init: B, mut f: F) -> Result<B, E2>
    where
        E2: From<E>,
        F: FnMut(B, T) -> Result<B, E2>,
    {
        self.0.try_rfold(init, |acc, v| f(acc, v?))
    }
}
struct IntoFallible<I>(I);
impl<T, I: DoubleEndedIterator<Item = T>> IntoFallible<I> {
    fn try_rfold<B, E2, F>(&mut self, init: B, f: F) -> Result<B, E2>
    where
        E2: From<Infallible>,
        F: FnMut(B, T) -> Result<B, E2>,
    {
        self.0.try_rfold(init, f)
    }
}

// itertools' shape: rfold over the Ok values, the first error from the back kept aside
fn process_results_rfold<I, T, E, B>(mut iter: I, error: &mut Result<(), E>, init: B, mut f: impl FnMut(B, T) -> B) -> B
where
    I: DoubleEndedIterator<Item = Result<T, E>>,
{
    iter.try_rfold(init, |acc, opt| match opt {
        Ok(x) => Ok(f(acc, x)),
        Err(e) => {
            *error = Err(e);
            Err(acc)
        }
    })
    .unwrap_or_else(|e| e)
}

#[derive(Debug, PartialEq)]
struct Wide(String);
impl From<u8> for Wide {
    fn from(code: u8) -> Wide {
        Wide(format!("code {code}"))
    }
}

#[test]
fn a01_try_rfold_stops_at_the_first_break_from_the_back() {
    let v = [1u32, 2, 3, 4, 5];
    // runs to the front: Some of the whole fold, in back-to-front order
    let mut it = v.iter();
    let order = it.try_rfold(String::new(), |mut acc, x| {
        acc.push_str(&x.to_string());
        Some(acc)
    });
    assert_eq!(order.as_deref(), Some("54321"));
    assert_eq!(it.next(), None);

    // stops at 3: the elements in front of it stay, 3 itself is consumed
    let mut it = v.iter();
    let r: Option<u32> = it.try_rfold(0, |acc, &x| if x == 3 { None } else { Some(acc + x) });
    assert_eq!(r, None);
    assert_eq!(it.as_slice(), &[1, 2]);
    assert_eq!(it.next_back(), Some(&2));

    // Result and ControlFlow carry their break value out
    let mut it = v.iter().copied();
    let r: Result<u32, String> = it.try_rfold(0, |acc, x| if acc + x > 9 { Err(format!("over at {x}")) } else { Ok(acc + x) });
    assert_eq!(r, Err("over at 3".to_string()));
    assert_eq!(it.collect::<Vec<_>>(), [1, 2]);
    let mut chars = "héllo wörld".chars();
    let cf = chars.try_rfold(0usize, |n, c| if c == ' ' { ControlFlow::Break(n) } else { ControlFlow::Continue(n + 1) });
    assert_eq!(cf, ControlFlow::Break(5));
    assert_eq!(chars.as_str(), "héllo");
    let all: ControlFlow<(), i64> = (1..=4i64).try_rfold(1, |p, x| ControlFlow::Continue(p * x));
    assert_eq!(all, ControlFlow::Continue(24));

    // an empty iterator gives the initial value back
    let none: Option<u8> = std::iter::empty::<u8>().try_rfold(7, |_, _| None);
    assert_eq!(none, Some(7));
}

#[test]
fn a02_through_adapters_and_borrowed_iterators() {
    // rev turns it round: try_rfold of a reversed iterator runs front to back
    let mut seen = Vec::new();
    let r: Option<()> = [10, 20, 30].iter().rev().try_rfold((), |(), &x| {
        seen.push(x);
        (x < 20).then_some(())
    });
    assert_eq!((r, &seen[..]), (None, &[10, 20][..]));

    // by_ref and a trait object: the caller keeps the iterator
    let mut range = 0..10u32;
    let hit = range.by_ref().try_rfold((), |(), x| if x % 4 == 0 { Err(x) } else { Ok(()) });
    assert_eq!(hit, Err(8));
    assert_eq!(range, 0..8);
    let mut boxed: Box<dyn DoubleEndedIterator<Item = u32>> = Box::new(vec![4u32, 5, 6].into_iter());
    let mut dynamic: &mut dyn DoubleEndedIterator<Item = u32> = &mut *boxed;
    let sum: Result<u32, ()> = (&mut dynamic).try_rfold(0, |a, x| if x == 4 { Err(()) } else { Ok(a + x) });
    assert_eq!(sum, Err(()));
    assert_eq!(boxed.next(), None);

    let mut zipped = [1, 2, 3].iter().zip("abc".bytes()).enumerate().map(|(i, (n, c))| (i, *n, c as char));
    let last_two: Result<Vec<char>, Vec<char>> = zipped.try_rfold(Vec::new(), |mut acc, (i, _, c)| {
        acc.push(c);
        if i == 1 { Err(acc) } else { Ok(acc) }
    });
    assert_eq!(last_two, Err(vec!['c', 'b']));
    assert_eq!(zipped.next(), Some((0, 1, 'a')));
}

#[test]
fn a03_fallible_iterator_and_process_results_shapes() {
    let items: Vec<Result<u32, u8>> = vec![Ok(1), Err(9), Ok(3), Ok(4)];
    let mut conv = Convert(items.iter().cloned());
    let r: Result<u32, Wide> = conv.try_rfold(0, |acc, v| Ok(acc + v));
    assert_eq!(r, Err(Wide("code 9".to_string())));
    assert_eq!(conv.0.next(), Some(Ok(1)));
    let mut ok = Convert(vec![Ok::<u32, u8>(2), Ok(5)].into_iter());
    assert_eq!(ok.try_rfold(String::new(), |acc, v| Ok::<_, Wide>(format!("{acc}{v}"))), Ok("52".to_string()));

    let mut plain = IntoFallible([1u8, 2, 3].into_iter());
    let r: Result<u8, Infallible> = plain.try_rfold(0, |acc, v| Ok(acc * 2 + v));
    assert_eq!(r, Ok(17));

    let mut error = Ok(());
    let folded = process_results_rfold(items.iter().cloned(), &mut error, Vec::new(), |mut acc, x| {
        acc.push(x);
        acc
    });
    assert_eq!((folded, error), (vec![4, 3], Err(9)));
    let mut error: Result<(), u8> = Ok(());
    let folded = process_results_rfold([Ok(1), Ok(2)].into_iter(), &mut error, 0, |acc, x: i32| acc * 10 + x);
    assert_eq!((folded, error), (21, Ok(())));
}
