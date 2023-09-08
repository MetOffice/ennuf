! *****************************COPYRIGHT*******************************
! (C) Crown copyright Met Office. All rights reserved.
! For further details please refer to the file COPYRIGHT.txt
! which you should have received as part of this distribution.
! *****************************COPYRIGHT*******************************

MODULE neural_net_mod
IMPLICIT NONE
CONTAINS
    SUBROUTINE dense(x_in, y_out, n_in, n_out, weights, biases, activation)

    IMPLICIT NONE

    INTEGER, PARAMETER                                    :: precision = 4
    INTEGER                                               :: i, j

    INTEGER, INTENT(in)                                   :: n_in, n_out
    REAL(kind = precision), DIMENSION(n_in),  INTENT(in)  :: x_in
    REAL(kind = precision), DIMENSION(n_out), INTENT(out) :: y_out
    REAL(kind = precision), DIMENSION(n_in, n_out)        :: weights
    REAL(kind = precision), DIMENSION(n_out)              :: biases

    REAL(kind = precision)                                :: running_sum
    REAL(kind = precision)                                :: denominator
    REAL(kind = precision), DIMENSION(n_out)              :: x
    REAL(kind = precision)                                :: alpha

    CHARACTER (LEN=10)                                    :: activation

    denominator = 0.0

    DO j=1,n_out
       running_sum = 0.0

       DO i=1,n_in
          running_sum = running_sum + ( x_in(i) * weights(i,j) )

       END DO
       x(j) = running_sum + biases(j)

       denominator = denominator + exp( x(j) )

    ENDDO

    DO j=1,n_out

       SELECT CASE (activation)
       CASE ("relu      ")
          y_out(j) = max(0.0,x(j))
       CASE ("leakyrelu ")
          alpha=0.1
          y_out(j) = max(alpha*x(j),x(j))
       CASE ("sigmoid   ")
          y_out(j) = 1.0 / ( 1.0 + exp(-x(j)) )
       CASE ("tanh      ")
          y_out(j) = tanh(x(j))
       CASE ("softmax   ")
          y_out(j) = exp(x(j)) / denominator
       CASE default
          PRINT*,'You have asked for an activation function that is not available.'
          PRINT*,'Please check your spelling or add it as an option.'
          y_out(j) = -99.999
       END SELECT

    ENDDO

    END SUBROUTINE dense

END MODULE neural_net_mod