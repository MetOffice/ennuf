! *****************************COPYRIGHT*******************************
! (C) Crown copyright Met Office. All rights reserved.
! For further details please refer to the file COPYRIGHT.txt
! which you should have received as part of this distribution.
! *****************************COPYRIGHT*******************************

MODULE svr_mod

IMPLICIT NONE

! ! INTEGER, PARAMETER :: precision=4
! ! REAL(kind = precision), DIMENSION(2)  :: y_out
! ! CALL svr(3,2,2,y_out, 4)
! INTEGER :: n_supp_vectors, n_dims, n_in
! REAL(kind = 4) :: intercept
! !real :: supp_vectors(n_supp_vectors,n_dims), dual_coef(n_supp_vectors), x_in(n_in, n_dims), y_out(n_in), temp(n_dims)
! REAL(kind = 4), DIMENSION(5,1) :: supp_vectors
! REAL(kind = 4), DIMENSION(5) :: dual_coef
! REAL(kind = 4), DIMENSION(1,1)  :: x_in
! REAL(kind = 4), DIMENSION(5)  :: temp
! REAL(kind = 4), DIMENSION(1)  :: y_out
! x_in=reshape((/ 0.38 /), shape(x_in))
! supp_vectors=reshape((/ -1.18184952, 0.2020686, -1.12780997, 1.16483002, -1.23727212 /), shape(supp_vectors))
! dual_coef=[-91.7370892,   -8.2629108,  100.0       ,  100.0       , -100.0]
! n_supp_vectors=5
! n_dims=1
! n_in=1
! ! y_out=[0]
! intercept=-5.39999999
! CALL svr(x_in, supp_vectors, dual_coef, n_supp_vectors, n_dims, n_in, y_out, intercept)

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