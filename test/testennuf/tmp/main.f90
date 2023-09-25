! Created by  on 18/09/2023.

PROGRAM main
    USE placeholder_mod, ONLY: placeholder
    IMPLICIT NONE
    INTEGER, PARAMETER :: precision=4
    REAL(KIND=precision) :: x(6)
    REAL(KIND=precision) :: scalars(6)
    REAL(KIND=precision) :: y_outputs_1(1)
    REAL(KIND=precision) :: y_outputs_2(2)
    INTEGER :: i, j, k, unit
    REAL(KIND=precision) :: example_array(3,2,5)
    ! initialise unit
    unit = 33
    ! initialise example array
    DO i = 1, 3
        DO j = 1, 2
            DO k = 1, 5
                example_array(i,j,k) = REAL(i + j + k, kind=precision)
            END DO
        END DO
    END DO
!    PRINT*, example_array
    ! What we want to do:
    ! Write the inputs to a file in python, in fortran-order.
    ! Read the inputs from that file, which should be easy because we know the shape.
    ! Write the outputs to a file, just 1d because again we don't care about their shape.
    ! Read that file in in python
!    CALL write_array_to_file(filename='testiomod.dat', array=example_array)
    ! Read array
    OPEN(unit, FILE="scalars.dat",&
     FORM="UNFORMATTED", STATUS="UNKNOWN", ACTION="READ", ACCESS='STREAM')
    READ(unit) x
    CLOSE(unit)
    PRINT*, 'arr:'
    PRINT*, x
    CALL placeholder(x, y_outputs_1, y_outputs_2)
    PRINT*, 'scalars:'
    PRINT*, x
    PRINT*, 'y_outputs_1:'
    PRINT*, y_outputs_1
    PRINT*, 'y_outputs_2:'
    PRINT*, y_outputs_2
END PROGRAM main