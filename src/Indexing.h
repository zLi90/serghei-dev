#ifndef _INDEXING_H_
#define _INDEXING_H_

#include "define.h"

// Unpack 2D indices
template <class I1, class I2> KOKKOS_INLINE_FUNCTION int packIndicesUniformGrid(I1 n1, I1 n2, I2 i1, I2 i2) {
  // i1 = j
  // i2 = i
  return(i2 + n2*i1);
}

// Unpack 2D indices
template <class I1, class I2, class I3> KOKKOS_INLINE_FUNCTION void unpackIndicesUniformGrid(I1 iGlob, I2 n1, I2 n2, I3 &i1, I3 &i2) {
  i1 = (iGlob/(n2))     ;
  i2 = (iGlob     ) % n2;
}


// Unpack 3D indices
template <class I1, class I2, class I3> KOKKOS_INLINE_FUNCTION void unpackIndicesUniformGrid(I1 iGlob, I2 n1, I2 n2, I2 n3, I3 &i1, I3 &i2, I3 &i3) {
  i1 = (iGlob/(n3*n2))     ;
  i2 = (iGlob/(n3   )) % n2;
  i3 = (iGlob        ) % n3;
}
/*


// Unpack 4D indices
template <class I1, class I2, class I3> KOKKOS_INLINE_FUNCTION void unpackIndicesUniformGrid(I1 iGlob, I2 n1, I2 n2, I2 n3, I2 n4, I3 &i1, I3 &i2, I3 &i3, I3 &i4) {
  i1 = (iGlob/(n4*n3*n2))     ;
  i2 = (iGlob/(n4*n3   )) % n2;
  i3 = (iGlob/(n4      )) % n3;
  i4 = (iGlob           ) % n4;
}


// Unpack 5D indices
template <class I1, class I2, class I3> KOKKOS_INLINE_FUNCTION void unpackIndicesUniformGrid(I1 iGlob, I2 n1, I2 n2, I2 n3, I2 n4, I2 n5, I3 &i1, I3 &i2, I3 &i3, I3 &i4, I3 &i5) {
  i1 = (iGlob/(n5*n4*n3*n2))     ;
  i2 = (iGlob/(n5*n4*n3   )) % n2;
  i3 = (iGlob/(n5*n4      )) % n3;
  i4 = (iGlob/(n5         )) % n4;
  i5 = (iGlob              ) % n5;
}


// Unpack 6D indices
template <class I1, class I2, class I3> KOKKOS_INLINE_FUNCTION void unpackIndicesUniformGrid(I1 iGlob, I2 n1, I2 n2, I2 n3, I2 n4, I2 n5, I2 n6, I3 &i1, I3 &i2, I3 &i3, I3 &i4, I3 &i5, I3 &i6) {
  i1 = (iGlob/(n6*n5*n4*n3*n2))     ;
  i2 = (iGlob/(n6*n5*n4*n3   )) % n2;
  i3 = (iGlob/(n6*n5*n4      )) % n3;
  i4 = (iGlob/(n6*n5         )) % n4;
  i5 = (iGlob/(n6            )) % n5;
  i6 = (iGlob                 ) % n6;
}
*/

#endif
