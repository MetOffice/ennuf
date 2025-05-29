! *****************************COPYRIGHT*******************************
! (C) Crown copyright Met Office. All rights reserved.
! For further details please refer to the file COPYRIGHT.txt
! which you should have received as part of this distribution.
! *****************************COPYRIGHT*******************************

MODULE svr_mod

IMPLICIT NONE

CONTAINS

   SUBROUTINE rbf(n_supp_vectors, n_dims, supp_vectors, x_in, y_out)
      IMPLICIT NONE

      INTEGER, PARAMETER                                    :: precision = 4
      INTEGER, INTENT(in)                                   :: n_supp_vectors, n_dims
      INTEGER                                               :: j
      REAL(kind = precision), DIMENSION(n_dims),  INTENT(in)  :: x_in
      REAL(kind = precision), DIMENSION(n_supp_vectors, n_dims), INTENT(in) :: supp_vectors
      REAL(kind = precision), DIMENSION(n_supp_vectors), INTENT(out) :: y_out
      REAL(kind = precision), DIMENSION(n_supp_vectors, n_dims) :: temp

      temp=SPREAD(x_in,1,n_supp_vectors)

      y_out = exp(-1*NORM2(supp_vectors-temp,2)**2)
   END SUBROUTINE rbf   


   SUBROUTINE svr(x_in, supp_vectors, dual_coef, n_supp_vectors, n_dims, n_in, y_out, intercept)
      IMPLICIT NONE

      INTEGER, PARAMETER                                    :: precision = 4
      INTEGER :: j
      INTEGER, INTENT(in) :: n_supp_vectors, n_dims, n_in
      REAL(kind = precision), INTENT(in) :: intercept
      ! real :: supp_vectors(n_supp_vectors,n_dims), dual_coef(n_supp_vectors), x_in(n_in, n_dims), y_out(n_in), temp(n_dims)
      REAL(kind = precision), DIMENSION(n_supp_vectors,n_dims), INTENT(in) :: supp_vectors
      REAL(kind = precision), DIMENSION(n_supp_vectors), INTENT(in) :: dual_coef
      REAL(kind = precision), DIMENSION(n_in,n_dims), INTENT(in)  :: x_in
      REAL(kind = precision), DIMENSION(n_supp_vectors)  :: temp
      REAL(kind = precision), DIMENSION(n_in),  INTENT(out)  :: y_out

      do j=1, size(x_in,1)
         CALL rbf(n_supp_vectors, n_dims, supp_vectors, x_in(j,:), temp)
         y_out(j) = DOT_PRODUCT(dual_coef, temp)+intercept
      end do
      
   END SUBROUTINE svr

END MODULE svr_mod
