! *****************************COPYRIGHT*******************************
! (C) Crown copyright Met Office. All rights reserved.
! For further details please refer to the file COPYRIGHT.txt
! which you should have received as part of this distribution.
! *****************************COPYRIGHT*******************************

module sample_multivariate_normal_mod
implicit none
contains
subroutine sample_multivariate_normal(mean, covariance_matrix, output_vector)
    ! Makes output_vector a sample from a multivariate normal distribution
    ! of the specified mean and covariance matrix.
    implicit none
    integer, parameter :: precision = 4
    ! arguments
    real(kind=precision), dimension(:), intent(in) :: mean
    real(kind=precision), dimension(:, :), intent(in) :: covariance_matrix
    real(kind=precision), dimension(:), intent(out) :: output_vector
    ! locals
    real(kind=precision), dimension(:), allocatable :: z
    real(kind=precision), dimension(:, :), allocatable :: decomp
    integer :: len_mean, i

    ! Initialise z to be filled with random normals
    len_mean = size(mean)
    allocate(z(len_mean))
    do i = 1, len_mean
        z(i) = rnorm()
    end do
    allocate(decomp(len_mean, len_mean))
    call cholesky_banachiewicz(covariance_matrix, decomp)
    output_vector = mean + matmul(decomp, z)
    deallocate(z)
    deallocate(decomp)
end subroutine sample_multivariate_normal

subroutine cholesky_banachiewicz(positive_definite, decomp)
    ! Implements the Cholesky-Banachiewicz algorithm for performing Cholesky decomposition.
    ! This is a decomposition of a Hermitian, positive-definite matrix into
    ! the product of a lower triangular matrix and its conjugate transpose,
    ! which is useful for efficient numerical solutions.
    implicit none
    ! constants
    integer, parameter :: precision = 4
    ! arguments
    real(kind=precision), dimension(:,:), intent(in) :: positive_definite
    real(kind=precision), dimension(:, :), intent(out) :: decomp
    ! locals
    real(kind=precision) :: sum
    integer i, j, k
    complex(kind=precision) :: aijminussigma, logdecomp
    real, parameter :: vsmall = tiny( 1.0 )

    do i = 1, size(positive_definite,1)
        do j = 1, i
            sum = 0
            do k = 1, j-1
                sum = sum + decomp(i,k) * decomp(j,k)
            end do
            if (i == j) then
                decomp(i, j) = sqrt(positive_definite(i,i) - sum)
            else
                if ((positive_definite(i, j) - sum) < vsmall) then
                    decomp(i, j) = 0
                else
                    aijminussigma = complex(positive_definite(i, j) - sum, 0)
                    logdecomp = log(aijminussigma) - log(decomp(j, j))
                    decomp(i, j) = real(exp(logdecomp))
                end if
            end if
        end do
    end do
end subroutine cholesky_banachiewicz

function rnorm() result( fn_val )
    !   Adapted from code released into the public domain by Alan Miller
    !   https://jblevins.org/mirror/amiller/rnorm.f90
    !   This version doubles the computations required for many calls
    !   but is thread-safe.
    !   Generate a random normal deviate using the polar method.
    !   Reference: Marsaglia,G. & Bray,T.A. 'A convenient method for generating
    !              normal variables', Siam Rev., vol.6, 260-264, 1964.

    implicit none
    real  :: fn_val

    ! Local variables
    real            :: u, v, sum, sln
    real, parameter :: one = 1.0, vsmall = tiny( one )

    ! Generate a pair of random normals
    do
    call random_number( u )
    call random_number( v )
    u = scale( u, 1 ) - one
    v = scale( v, 1 ) - one
    sum = u*u + v*v + vsmall         ! vsmall added to prevent LOG(zero) / zero
    if (sum < one) exit
    end do
    sln = sqrt(- scale( log(sum), 1 ) / sum)
    fn_val = u*sln
return
end function rnorm
end module sample_multivariate_normal_mod