use std::collections::BinaryHeap;

pub trait Stack<T>
where
    T: Ord,
{
    fn stack_push(&mut self, _: T);
    fn stack_pop(&mut self) -> Option<T>;
}

impl<T> Stack<T> for Vec<T>
where
    T: Ord,
{
    #[inline(always)]
    fn stack_push(&mut self, element: T) {
        Vec::<T>::push(self, element)
    }
    #[inline(always)]
    fn stack_pop(&mut self) -> Option<T> {
        Vec::<T>::pop(self)
    }
}

impl<T> Stack<T> for BinaryHeap<T>
where
    T: Ord,
{
    #[inline(always)]
    fn stack_push(&mut self, element: T) {
        BinaryHeap::<T>::push(self, element)
    }
    #[inline(always)]
    fn stack_pop(&mut self) -> Option<T> {
        BinaryHeap::<T>::pop(self)
    }
}
