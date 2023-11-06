! *****************************COPYRIGHT*******************************
! (C) Crown copyright Met Office. All rights reserved.
! For further details please refer to the file COPYRIGHT.txt
! which you should have received as part of this distribution.
! *****************************COPYRIGHT*******************************

MODULE neural_net_mod
! Contains subroutines for 1d dense layers, plus layers common in CNNs:
! pooling1d, conv1d, pixelshuffle1d, skip connection, dense3d (which takes and returns 3d inputs but is otherwise
! just a regular dense layer)
IMPLICIT NONE
CONTAINS

    SUBROUTINE dense(x_in, y_out, n_in, n_out, weights, biases, activation, alpha)

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
    REAL(kind = precision), INTENT(in)                    :: alpha

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
       CASE ("linear    ")
           y_out(j) = x(j)
       CASE ("leakyrelu ")
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

    SUBROUTINE concatenate_1d(x_in_1, x_in_2, y_out)
        IMPLICIT NONE
        INTEGER, PARAMETER :: precision = 4
        REAL(KIND=precision), DIMENSION(:), INTENT(IN) :: x_in_1, x_in_2
        REAL(KIND=precision), DIMENSION(:), INTENT(OUT) :: y_out
        INTEGER :: len1, len2, total_len
        len1 = SIZE(x_in_1)
        len2 = SIZE(x_in_2)
        total_len = len1 + len2
        IF (SIZE(y_out) /= total_len) THEN
            PRINT*, 'Arr y_out had size:'
            PRINT*, SIZE(y_out)
            PRINT*, 'but was expecting:'
            PRINT*, total_len
            PRINT*, 'exiting...'
            CALL EXIT(1)
        END IF
        y_out(1:len1) = x_in_1
        y_out(len1+1:total_len) = x_in_2
    END SUBROUTINE concatenate_1d

    SUBROUTINE conv_1d( &
    ! arrays for input and output
    input, &
    output, &
    ! dimensions of the arrays
    number_samples, &
    channels_in, &
    channels_out, &
    length_in, &
    length_out, &
    ! dims and values of the kernels and biases
    size_kernel, &
    kernels, &
    biases, &
    ! optional arguments: padding, stide and dilation
    arg_padding, &
    arg_stride, &
    arg_dilation)

    IMPLICIT NONE
    !
    ! Purpose:
    !   This subroutine performs a 1D convolution for
    !   use in ML applications in the models

    ! Method:
    !   The subroutine gets as input 1D signals and the
    !   values of all the weitghts and biases and then
    !   computes the output from them
    !

    ! Subroutine Arguments:-----------------------------------------------

    INTEGER, PARAMETER :: precision = 4

    ! Dimensions of arrays
    INTEGER, INTENT(in) :: number_samples, channels_in, channels_out, length_in, length_out, size_kernel

    ! Input and output arrays
    REAL(kind = precision), DIMENSION(number_samples, channels_in, length_in),    INTENT(in) :: input
    REAL(kind = precision), DIMENSION(number_samples, channels_out, length_out), INTENT(out) :: output

    ! Weights and biases
    REAL(kind = precision), DIMENSION(channels_out, channels_in, size_kernel)                :: kernels
    REAL(kind = precision), DIMENSION(channels_out)                                          :: biases

    ! Optional arguments (padding stride and dilation)
    INTEGER, OPTIONAL, INTENT(in) :: arg_padding , arg_stride, arg_dilation
    INTEGER                       :: padding , stride, dilation

    ! Auxiliary variables
    INTEGER :: n, c_in, c_out, l_k, l_out, s

    ! ---------------------------------------------------
    ! Atrribute padding dilation and stride values from
    ! arguments if present or set to default otherwise
    ! ---------------------------------------------------

    IF(PRESENT(arg_padding)) THEN
       padding = arg_padding
    ELSE
       padding = 0   ! default value
    END IF

    IF(PRESENT(arg_stride)) THEN
       stride = arg_stride
    ELSE
       stride = 1   ! default value
    END IF

    IF(PRESENT(arg_dilation)) THEN
       dilation = arg_dilation
    ELSE
       dilation = 1   ! default value
    END IF

    ! -------------------------------------------------
    ! Check for consistency in array dimensions
    ! -------------------------------------------------

    IF( length_out /= INT( 1 + ( length_in + 2 * padding - dilation * (size_kernel - 1) - 1) / stride ) ) THEN
        PRINT*, "ERROR: "
        PRINT*, "The dimensions of the output array do not correpond to the expected"
        PRINT*, "Check the values of padding, stride and dilation"
        PRINT*, "Ensure that length_out = 1 + ( length_in + 2 * padding - dilation * (size_kernel - 1) - 1) / stride"
        CALL EXIT(1)
    END IF

    ! Main block: --------------------------------------
    ! Loop round samples to perform convolution on each
    ! --------------------------------------------------

    DO n=1, number_samples
         DO c_out=1, channels_out
            DO c_in=1, channels_in
                s= 0
                DO l_out=1, length_out
                    DO l_k=1, size_kernel
                        output(n, c_out,l_out) = output(n, c_out,l_out) + kernels(c_out,c_in,l_k)*input(n,c_in, s+l_k)
                    END DO
                s = s + stride
                END DO
            END DO
            output(n, c_out, :) = output(n, c_out, :) + biases(c_out)
        END DO
     END DO


    END SUBROUTINE conv_1d

!----------------------------------------------------------------------------------------------------------

    SUBROUTINE pooling_1d( &
    ! arrays for input and output
    input, &
    output, &
    ! dimensions of the arrays
    number_samples, &
    channels, &
    length_in, &
    length_out, &
    ! type and scale of pooling
    choice_of_pooling, &
    pool_size, &
    ! optional arguments (padding, stride, dilation)
    arg_padding, &
    arg_stride, &
    arg_dilation)

    IMPLICIT NONE
    !
    ! Purpose:
    !   This subroutine performs 1D  pooling for
    !   use in ML applications in the models

    ! Method:
    !   The subroutine gets as input 1D signals and then
    !   computes the output from them according to the
    !   choice of pooling (average or maximum)
    !

    ! Subroutine Arguments:-----------------------------------------------

    INTEGER, PARAMETER :: precision = 4

    ! Dimensions of arrays (input and output)
    INTEGER, INTENT(in) :: number_samples, channels, length_in, length_out

    ! Input and output arrays
    REAL(kind = precision), DIMENSION(number_samples, channels, length_in), INTENT(in)   :: input
    REAL(kind = precision), DIMENSION(number_samples, channels, length_out), INTENT(out) :: output

    ! Pooling choices
    CHARACTER(LEN=3)    :: choice_of_pooling
    INTEGER, INTENT(in) :: pool_size

    ! Optional arguments (padding, stride, dilation)
    INTEGER, OPTIONAL, INTENT(in) :: arg_padding , arg_stride, arg_dilation
    INTEGER                       :: padding , stride, dilation

    ! Auxiliary variables
    INTEGER :: n,c,l, s

    ! ---------------------------------------------------
    ! Atrribute padding, dilation and stride values from
    ! arguments if present or set to default otherwise
    ! ---------------------------------------------------

    IF(PRESENT(arg_padding)) THEN
        padding = arg_padding
    ELSE
        padding = 0   ! default value
    END IF

    IF(PRESENT(arg_stride)) THEN
        stride = arg_stride
    ELSE
        stride = pool_size  ! default value
    END IF

    IF(PRESENT(arg_dilation)) THEN
        dilation = arg_dilation
    ELSE
        dilation = 1   ! default value
    END IF

    ! -------------------------------------------------
    ! Check for consistency in array dimensions
    ! -------------------------------------------------

    IF( length_out /= INT( 1 + ( length_in + 2 * padding - dilation * (pool_size - 1) - 1) / stride )  ) THEN
        PRINT*, "ERROR:"
        PRINT*, "The dimensions of the output array do not correpond to the expected"
        PRINT*, "Check the values of padding, stride and dilation"
        PRINT*, "Ensure that length_out = 1 + ( length_in + 2 * padding - dilation * (pool_size - 1) - 1) / stride "
        CALL EXIT(1)
    END IF

    ! Main block: --------------------------------------
    ! Select choice of pooling and loop round samples
    ! to perform pooling on the data
    ! --------------------------------------------------

    SELECT CASE (choice_of_pooling)

    CASE ("MAX")
    ! MAX POOLING outputs the maximum value within the
    ! pooling window while it slides through the data

        DO n=1, number_samples
            DO c=1, channels
                s = 1
                DO l=1, length_out
                        output(n, c, l) = MAXVAL(input(n, c, s : s + pool_size - 1))
                    s = s + stride
                END DO
            END DO
        END DO

    CASE ("AVG")
    ! AVG POOLING outputs the average of the values within
    ! the pooling window while it slides through the data

        DO n=1, number_samples
            DO c=1, channels
                s = 1
                DO l=1, length_out
                    output(n, c, l) = SUM(input(n, c, s : s + pool_size - 1)) / pool_size
                    s = s + stride
                END DO
            END DO
        END DO

    CASE default
        PRINT*, "ERROR: "
        PRINT*,'You have asked for a choice of pooling that is not available.'
        PRINT*,'The currently available choices of pooling are "MAX" and "AVG".'
        PRINT*,'Please check your spelling or add it as an option.'
        CALL EXIT(1)
    END SELECT

    END SUBROUTINE pooling_1d

!------------------------------------------------------------------------------------

    SUBROUTINE activation_function(&
    ! array with data
    input, &
    output, &
    ! dimensions of array
    number_samples, &
    channels, &
    length, &
    ! choice of activation function
    activation, &
    ! optional argument (alpha for leaky relu)
    arg_alpha)

    IMPLICIT NONE
    !
    ! Purpose:
    !   This subroutine apllies a chosen activation function
    !   to the input, for use in ML applications in the models

    ! Method:
    !   The subroutine gets as input 1D signals and then
    !   computes the output from them depending on the
    !   activation function chosen by the user
    !

    ! Subroutine Arguments:-----------------------------------------------

    INTEGER, PARAMETER :: precision = 4

    ! Dimensions of the data array
    INTEGER, INTENT(in)  :: number_samples,channels, length
    ! Array with the data
    REAL(kind = precision), DIMENSION(number_samples, channels, length) :: input
    REAL(kind = precision), DIMENSION(number_samples, channels, length) :: output

    ! Choice of Activation Funtion
    CHARACTER (LEN=10) :: activation

    ! Optional argument (negative slope for the Leaky ReLU)
    REAL(kind = precision), OPTIONAL, INTENT(in) :: arg_alpha
    REAL(kind = precision)                       :: alpha

    ! Auxiliary variables
    INTEGER :: n,c,l

    ! -------------------------------------------------
    ! Warning message for inconsisency in arguments
    ! -------------------------------------------------

    IF(PRESENT(arg_alpha)) THEN
        IF(activation /= "leakyrelu ") THEN
            PRINT*, "WARNING: "
            PRINT*, "The activation function you chose does not take alpha as an argument."
            PRINT*, "alpha is the negative slope for the Leaky ReLU"
        END IF
    END IF

    ! Main block: --------------------------------------
    ! Select choice of activation function and loop
    ! round samples to calculate the output
    ! --------------------------------------------------

    SELECT CASE (activation)

    CASE ("relu      ")

        DO n=1,number_samples
            DO c=1, channels
                DO l=1, length
                    output(n,c,l) = max(0.0, input(n,c,l))
                END DO
            END DO
        END DO

    CASE ("leakyrelu ")

        IF(PRESENT(arg_alpha)) THEN
            alpha = arg_alpha
        ELSE
            alpha = 0.01 ! default value
        END IF

        DO n=1,number_samples
            DO c=1, channels
                DO l=1, length
                    output(n,c,l) = max(alpha*input(n,c,l),input(n,c,l))
                END DO
            END DO
        END DO

    CASE ("sigmoid   ")

        DO n=1,number_samples
            DO c=1, channels
                DO l=1, length
                    output(n,c,l) = 1.0 / ( 1.0 + exp(-input(n,c,l)) )
                END DO
            END DO
        END DO

    CASE ("tanh      ")

        DO n=1,number_samples
            DO c=1, channels
                DO l=1, length
                    output(n,c,l) = tanh(input(n,c,l))
                END DO
            END DO
        END DO

    CASE ("softmax   ")

        DO n=1,number_samples
            DO c=1, channels
                DO l=1, length
                    output(n,c,l) = exp(input(n,c,l)) / SUM( exp(input(n,c,:)) )
                END DO
            END DO
        END DO

    CASE default
        PRINT*, "ERROR: "
        PRINT*,'You have asked for an activation function that is not available.'
        PRINT*,'Please check your spelling or add it as an option.'
        PRINT*,'Currently available functions are: ReLU, Leaky ReLU, Sigmoid, Tanh and Softmax.'
        CALL EXIT(1)

    END SELECT

    END SUBROUTINE activation_function

!--------------------------------------------------------------------------------------------------

    SUBROUTINE pixel_shuffle_1d(&
    ! arrays for input and output
    input, &
    output, &
    ! dimensions of data
    number_samples, &
    channels_in, &
    channels_out, &
    length_in, &
    length_out, &
    ! upscale factor
    upscale_factor)

    IMPLICIT NONE
    !
    ! Purpose:
    !   This subroutine performa a 1D Pixel Shuffle,
    !   for use in ML applications in the models

    ! Method:
    !   The subroutine gets as input 1D signals and
    !   upsamples them by the factor chosen by the user
    !

    ! Subroutine Arguments:-----------------------------------------------

    INTEGER, PARAMETER :: precision = 4

    ! Dimensions of arrays
    INTEGER, INTENT(in) :: number_samples, channels_in, length_in, channels_out, length_out

    ! Data arrays
    REAL(kind=precision), DIMENSION(number_samples, channels_in, length_in)    :: input
    REAL(kind=precision), DIMENSION(number_samples, channels_out, length_out)  :: output

    ! Upscale factor
    INTEGER, INTENT(in) :: upscale_factor

    ! Intermediate arrays
    REAL(kind=precision), DIMENSION(number_samples, upscale_factor, channels_out, length_in) :: i_1
    REAL(kind=precision), DIMENSION(number_samples, channels_out, length_in, upscale_factor) :: i_2

    ! Auxiliary variables
    INTEGER :: n,c,u,j,l

    ! -------------------------------------------------
    ! Check for consistency in array dimensions
    ! -------------------------------------------------

    IF( channels_out /= INT(channels_in / upscale_factor) ) THEN
        PRINT*, "ERROR: "
        PRINT*, "The dimensions of the output do not correspond to expected"
        PRINT*, "Ensure that channels_out = channels_in / upscale_factor"
        CALL EXIT(1)
    ELSE IF ( length_out /= INT(length_in * upscale_factor) ) THEN
        PRINT*, "ERROR: "
        PRINT*, "The dimensions of the output do not correspond to expected"
        PRINT*, "Ensure that length_out = length_in * upscale_factor"
        CALL EXIT(1)
    END IF

    ! Main block: --------------------------------------
    ! Transform the data with the help of two auxiliary
    ! arrays by looping round all the samples
    ! --------------------------------------------------

    DO n=1, number_samples
        j=1
        DO u=1, upscale_factor
            i_1(:,u,:,:) = input(:,j:j+channels_out,:)
            j = j + channels_out
        END DO
    END DO

    DO n=1, number_samples
        DO c=1, channels_out
            DO u=1, upscale_factor
                i_2(n,c,:,u) = i_1(n,u,c,:)
            END DO
        END DO
    END DO

    DO n=1, number_samples
        DO c=1, channels_out
            j=1
            DO l=1, length_in
                output(n,c,j:j+upscale_factor) = i_2(n,c,l,:)
                j = j + upscale_factor
            END DO
        END DO
    END DO

    END SUBROUTINE pixel_shuffle_1d

! --------------------------------------------------------------------------------------------------

    SUBROUTINE dense_3d( &
    ! data arrays (input and output)
    input, &
    output, &
    ! dimensions of data
    number_samples, &
    channels, &
    length_in, &
    length_out, &
    ! values of the weights and biases
    weights, &
    biases)

    IMPLICIT NONE
    !
    ! Purpose:
    !   This subroutine performs a Linear transformation
    !   on the data, for use in ML applications in the models

    ! Method:
    !   The subroutine gets as input 1D signals and the weights
    !   and biases of the layer and ouputs the result from them
    !

    ! Subroutine Arguments:-----------------------------------------------

    INTEGER, PARAMETER :: precision = 4

    ! Dimensions of the arrays containing the data
    INTEGER, INTENT(in) :: number_samples, channels, length_in, length_out

    ! Arrays of data
    REAL(kind = precision), DIMENSION(number_samples, channels, length_in),  INTENT(in)   :: input
    REAL(kind = precision), DIMENSION(number_samples, channels, length_out), INTENT(out)  :: output

    ! Weights and biases
    REAL(kind = precision), DIMENSION(length_out, length_in)                              :: weights
    REAL(kind = precision), DIMENSION(length_out )                                        :: biases

    ! Auxiliary variables
    INTEGER :: n,c,l_out,l_in


    ! Main block: ----------------------------------------------
    ! Loop round samples and channels to calculate linear output
    ! ----------------------------------------------------------

    DO n=1,number_samples
        DO c=1,channels
            DO l_out=1, length_out
                DO l_in=1, length_in
                    output(n,c,l_out) = output(n,c,l_out) + ( input(n,c,l_in) * weights(l_out, l_in) + biases(l_out))
                END DO
            END DO
       END DO
    END DO

    END SUBROUTINE dense_3d

! -----------------------------------------------------------------------------------------------------

    SUBROUTINE skip_connection( &
    ! data arrays (inputs and output)
    input_1, &
    input_2, &
    output, &
    ! dimensions of input arrays
    number_samples, &
    channels, &
    length)

    IMPLICIT NONE
    !
    ! Purpose:
    !   This subroutine acts as a Skip Connection
    !   for use in ML applications in the models

    ! Method:
    !   The subroutine gets as input two sets of
    !   1D signals and concatentes them together

    ! Subroutine Arguments:-----------------------------------------------

    INTEGER, PARAMETER :: precision = 4

    ! Dimensions of input arrays
    INTEGER, INTENT(in) :: number_samples, channels, length

    ! Input arrays of data
    REAL(kind=precision), DIMENSION(number_samples, channels, length) :: input_1
    REAL(kind=precision), DIMENSION(number_samples, channels, length) :: input_2

    ! Output array of data
    REAL(kind=precision), DIMENSION(number_samples, INT(channels * 2 ), length) :: output

    ! Auxiliary variables
    INTEGER :: n,c

    ! Main block: ----------------------------------------------
    ! Loop round samples and concatenate the two
    ! input arraystogether into the ouput array
    ! ----------------------------------------------------------

    DO n=1, number_samples
        DO c=1, channels
            output(n,c,:) = input_1(n,c,:)
            output(n,c+channels,:) = input_2(n,c,:)
        END DO
    END DO

    END SUBROUTINE skip_connection

END MODULE neural_net_mod