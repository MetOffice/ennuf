! *****************************COPYRIGHT*******************************
! (C) Crown copyright Met Office. All rights reserved.
! For further details please refer to the file COPYRIGHT.txt
! which you should have received as part of this distribution.
! *****************************COPYRIGHT*******************************

MODULE neural_net_mod
! Contains subroutines for 1d dense layers, plus layers common in 1D FCNNs and CNNs:
  ! pooling, conv, skip connection, dense and activation functions
  
IMPLICIT NONE
CONTAINS

    SUBROUTINE dense( &
    ! data arrays (input and output)
    x_in, &
    y_out, &
    ! dimensions of data
    channels, &
    length_in, &
    length_out, &
    ! values of the weights and biases
    weights, &
    biases)

    ! I need to change things
    IMPLICIT NONE

    INTEGER, PARAMETER :: precision = 4

    ! Dimensions of the arrays containing the data
    INTEGER, INTENT(IN) :: &
      channels  &
    , length_in &
    , length_out

    ! Arrays of data
    REAL(kind = precision), INTENT(IN)   :: &
    x_in(channels, length_in)
    REAL(kind = precision), INTENT(OUT)  :: &
    y_out(channels, length_out)

    ! Weights and biases
    REAL(kind = precision), INTENT(IN) :: &
      weights(length_out, length_in) &
    , biases(length_out)

    ! Auxiliary variables
    INTEGER :: c,l_out,l_in


    DO c=1,channels
        DO l_out=1, length_out
            DO l_in=1, length_in
                y_out(c,l_out) = x_in(c,l_in) * weights(l_out, l_in)
            END DO
            y_out(c,l_out) = y_out(c,l_out) + biases(l_out)
        END DO
    END DO

    END SUBROUTINE dense

!-----------------------------------------------------------------------------------------------------
    SUBROUTINE conv_1d( &
    ! arrays for input and output
    x_in, &
    y_out, &
    ! dimensions of the arrays
    channels_in, &
    channels_out, &
    length_in, &
    length_out, &
    ! dims and values of the kernels and biases
    size_kernel, &
    weights, &
    biases, &
    ! padding, stide and dilation
    pad_mode, &
    padding, &
    stride, &
    dilation)

    IMPLICIT NONE

    INTEGER, PARAMETER :: precision = 4

    ! Dimensions of arrays
    INTEGER, INTENT(IN) :: &
      channels_in &
    , channels_out &
    , length_in &
    , length_out &
    , size_kernel

    ! Input and output arrays
    REAL(kind = precision),   INTENT(IN) :: &
    x_in(channels_in, length_in)
    REAL(kind = precision), INTENT(OUT) :: &
    y_out(channels_out, length_out)

    ! Weights and biases
    REAL(kind = precision), INTENT(IN) :: &
      weights(channels_out, channels_in, size_kernel) &
    , biases(channels_out)

    ! padding stride and dilation
    INTEGER, INTENT(IN) :: &
      padding &
    , stride &
    , dilation

	CHARACTER (LEN=7), INTENT(IN) :: &
	pad_mode 

    ! Intermediate array 
    INTEGER :: &
	length_inter
    REAL(kind = precision), ALLOCATABLE :: &
    inter(:,:)
    
    ! Auxiliary variables
    INTEGER :: c_in, c_out, l_out, l_k, s


    IF( length_out /= INT( 1 + ( length_in + 2 * padding - dilation * (size_kernel - 1) - 1) / stride ) ) THEN
        PRINT*, "ERROR: "
        PRINT*, "The dimensions of the output array do not correpond to the expected"
        PRINT*, "Check the values of padding, stride and dilation"
        PRINT*, "Ensure that length_out = 1 + ( length_in + 2 * padding - dilation * (size_kernel - 1) - 1) / stride"
        CALL EXIT(1)
    END IF

    length_inter = length_in + 2 * padding   
    ALLOCATE(inter(channels_in, length_inter))
   
	SELECT CASE (pad_mode)
	    CASE ("none   ")
	        inter = x_in
	    CASE ("zeros  ")
	        inter = 0.0
	        DO c_in=1, channels_in
	            inter(c_in,padding+1:-padding-1) = x_in(c_in,:)
	        END DO
	    CASE ("reflect")
	        DO c_in=1, channels_in
	            inter(c_in,padding+1:-padding-1) = x_in(c_in,:)
	            inter(c_in,:padding) = x_in(c_in,padding+1:2:-1)
	            inter(c_in,length_inter-padding+1:) = x_in(c_in,length_in-1:length_in-padding-1:-1)
	        END DO
	END SELECT


    DO c_out=1, channels_out
        DO c_in=1, channels_in
            s=0
            DO l_out=1, length_out
                DO l_k=1, size_kernel
                    y_out(c_out,l_out) = y_out(c_out,l_out) + weights(c_out,c_in,l_k) * inter(c_in, s+(l_k*dilation)-1)
                END DO
            s = s + stride
            END DO
        END DO
        y_out(c_out, :) = y_out(c_out, :) + biases(c_out)
    END DO

    IF ( ALLOCATED( inter         ) ) DEALLOCATE ( inter         )

    END SUBROUTINE conv_1d

!----------------------------------------------------------------------------------------------------------

    SUBROUTINE pooling_1d( &
    ! arrays for input and output
    x_in, &
    y_out, &
    ! dimensions of the arrays
    channels, &
    length_in, &
    length_out, &
    ! type and scale of pooling
    choice_of_pooling, &
    pool_size, &
    ! padding, stride, dilation
    padding, &
    stride)

    IMPLICIT NONE

    INTEGER, PARAMETER :: precision = 4

    ! Dimensions of arrays
    INTEGER, INTENT(IN) :: &
      channels &
    , length_in &
    , length_out
    
    ! Input and output arrays
    REAL(kind = precision),   INTENT(IN) :: &
    x_in(channels, length_in)
    REAL(kind = precision), INTENT(OUT) :: &
    y_out(channels, length_out)

    ! padding and stride 
    INTEGER, INTENT(IN) :: &
      padding &
    , stride 

    ! Pooling choices
    CHARACTER(LEN=3)    :: &
    choice_of_pooling
    INTEGER, INTENT(IN) :: &
    pool_size

    ! Intermediate array 
    INTEGER :: &
	length_inter
    REAL(kind = precision), ALLOCATABLE :: &
    inter(:,:)
    
    ! Auxiliary variables
    INTEGER :: c,l, s


    IF( length_out /= INT( 1 + ( length_in + 2 * padding - pool_size) / stride )  ) THEN
        PRINT*, "ERROR:"
        PRINT*, "The dimensions of the output array do not correpond to the expected"
        PRINT*, "Check the values of padding, stride and dilation"
        PRINT*, "Ensure that length_out = 1 + ( length_in + 2 * padding - pool_size ) / stride "
        CALL EXIT(1)
    END IF


    length_inter = length_in + 2 * padding   
    ALLOCATE(inter(channels, length_inter))
   
    
    SELECT CASE (choice_of_pooling)

    CASE ("MAX")

        inter = -9999.99
	    DO c=1, channels
	        inter(c,padding+1:-padding-1) = x_in(c,:)
	    END DO
        
        DO c=1, channels
            s = 1
            DO l=1, length_out
                y_out(c, l) = MAXVAL(inter(c, s : s + pool_size - 1))
                s = s + stride
            END DO
        END DO

    CASE ("AVG")

        inter = 0.0
	    DO c=1, channels
	        inter(c,padding+1:-padding-1) = x_in(c,:)
	    END DO
            
        DO c=1, channels
            s = 1
            DO l=1, length_out
                y_out(c, l) = SUM(inter(c, s : s + pool_size - 1)) / pool_size
                s = s + stride
            END DO
        END DO

    CASE default
        PRINT*, "ERROR: "
        PRINT*,'You have asked for a choice of pooling that is not available.'
        PRINT*,'The currently available choices of pooling are "MAX" and "AVG".'
        PRINT*,'Please check your spelling or add it as an option.'
        CALL EXIT(1)
    END SELECT

    IF ( ALLOCATED( inter         ) ) DEALLOCATE ( inter         )

    END SUBROUTINE pooling_1d

!------------------------------------------------------------------------------------

    SUBROUTINE activation_function(&
    ! arrays with data
    x_in, &
    y_out, &
    ! dimensions of arrays
    channels, &
    length, &
    ! choice of activation function
    activation, &
    ! optional argument (alpha for leaky relu)
    alpha)

    IMPLICIT NONE

    INTEGER, PARAMETER :: precision = 4

    ! Dimensions of the data array
    INTEGER, INTENT(IN)  :: &
      channels
    , length
    
    ! Array with the data
    REAL(kind = precision),   INTENT(IN) :: &
     x_in(channels, length)
    REAL(kind = precision), INTENT(OUT) :: &
     y_out(channels, length)

    ! Choice of Activation Funtion
    CHARACTER (LEN=10), INTENT(IN) :: &
     activation

    ! Optional argument (negative slope for the Leaky ReLU)
    REAL(kind = precision), OPTIONAL, INTENT(IN) ::
     alpha


    INTEGER :: &
     c,l

    IF(PRESENT(arg_alpha)) THEN
        IF(activation /= "leakyrelu ") THEN
            PRINT*, "WARNING: "
            PRINT*, "The activation function you chose does not take alpha as an argument."
            PRINT*, "alpha is the negative slope for the Leaky ReLU"
        END IF
    END IF
     
    IF(activation == "leakyrelu ") THEN
        IF(.NOT. PRESENT(arg_alpha)) THEN
            PRINT*, "WARNING: "
            PRINT*, "The activation function you chose takes alpha as an argument but no alpha was found."
        END IF
    END IF

    SELECT CASE (activation)

    CASE ("relu      ")

       DO c=1, channels
          DO l=1, length
             y_out(c,l) = max(0.0, x_in(c,l))
          END DO
       END DO

    CASE ("leakyrelu ")

        DO c=1, channels
           DO l=1, length
              y_out(c,l) = max(alpha*x_in(c,l), x_in(c,l))
           END DO
        END DO

    CASE ("sigmoid   ")

        DO c=1, channels
            DO l=1, length
                y_out(c,l) = 1.0 / ( 1.0 + exp(-x_in(c,l)) )
            END DO
        END DO

    CASE ("tanh      ")

        DO c=1, channels
           DO l=1, length
              y_out(c,l) = tanh(x_in(c,l))
           END DO
        END DO

    CASE ("softmax   ")

        DO c=1, channels
            DO l=1, length
               y_out(c,l) = exp(x_in(c,l)) / SUM( exp(x_in(c,:)) )
            END DO
        END DO

    CASE default
        PRINT*, "ERROR: "
        PRINT*,'You have asked for an activation function that is not available.'
        PRINT*,'Please check your spelling or add it as an option.'
        PRINT*,'Currently available functions are: relu, leakyrelu, sigmoid, tanh and softmax.'
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


!------------------------------------------------------------------------------------------------------


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
