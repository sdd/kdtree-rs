pub(crate) mod arrays {
    use core::option::Option::None;
    use serde::{
        de::{SeqAccess, Visitor},
        Deserialize,
        Deserializer, ser::SerializeTuple, Serialize, Serializer,
    };
    use std::{convert::TryInto, marker::PhantomData};

    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
        data: &[T; N],
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let mut s = ser.serialize_tuple(N)?;
        for item in data {
            s.serialize_element(item)?;
        }
        s.end()
    }

    struct ArrayVisitor<T, const N: usize>(PhantomData<T>);

    impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<T, N>
        where
            T: Deserialize<'de>,
    {
        type Value = [T; N];

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str(&format!("an array of length {}", N))
        }

        #[inline]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
        {
            // can be optimized using MaybeUninit
            let mut data = Vec::with_capacity(N);
            for _ in 0..N {
                match (seq.next_element())? {
                    Some(val) => data.push(val),
                    None => return Err(serde::de::Error::invalid_length(N, &self)),
                }
            }
            match data.try_into() {
                Ok(arr) => Ok(arr),
                Err(_) => unreachable!(),
            }
        }
    }
    pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<[T; N], D::Error>
        where
            D: Deserializer<'de>,
            T: Deserialize<'de>,
    {
        deserializer.deserialize_tuple(N, ArrayVisitor::<T, N>(PhantomData))
    }
}

pub(crate) mod vec_arrays {
    use core::option::Option::None;
    use serde::{
        de::{SeqAccess, Visitor},
        Deserialize, Deserializer, Serialize, Serializer,
    };
    use serde::ser::SerializeSeq;
    use std::{convert::TryInto, marker::PhantomData};

    pub fn serialize<S: Serializer, T: Serialize, const N: usize>(
        data: &Vec<[T; N]>,
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        let mut s = ser.serialize_seq(Some(data.len() * N))?;
        for point in data.iter() {
            for item in point {
                s.serialize_element(item)?;
            }
        }

        s.end()
    }

    struct VecArrayVisitor<T, const N: usize>(PhantomData<T>);

    impl<'de, T, const N: usize> Visitor<'de> for VecArrayVisitor<T, N>
        where
            T: Deserialize<'de>,
    {
        type Value = Vec<[T; N]>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str(&format!("a vector of arrays of length {}", N))
        }

        #[inline]
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
        {
            // can be optimized using MaybeUninit
            let mut result = if let Some(len) = seq.size_hint() {
                Vec::with_capacity(len / N)
            } else {
                Vec::new()
            };

            while let Some(val) = seq.next_element()? {
                let mut item = Vec::with_capacity(N);
                item.push(val);

                for _ in 1..N {
                    match (seq.next_element())? {
                        Some(val) => item.push(val),
                        None => return Err(serde::de::Error::invalid_length(N, &self)),
                    }
                }

                let item_arr: [T; N] = match item.try_into() {
                    Ok(arr) => Ok(arr),
                    Err(_) => unreachable!(),
                }?;

                result.push(item_arr);
            }

            Ok(result)
        }
    }

    pub fn deserialize<'de, D, T, const N: usize>(deserializer: D) -> Result<Vec<[T; N]>, D::Error>
        where
            D: Deserializer<'de>,
            T: Deserialize<'de>,
    {
        deserializer.deserialize_seq(VecArrayVisitor::<T, N>(PhantomData))
    }
}
