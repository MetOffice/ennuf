! Adapted from https://stackoverflow.com/questions/8828377/reading-data-from-txt-file-in-fortran

MODULE matrix_txt_reader_mod
IMPLICIT NONE
CONTAINS
FUNCTION txt_to_r_array(filename) RESULT(x)
    IMPLICIT NONE

    CHARACTER(*), INTENT(IN) :: filename

    CHARACTER(128) :: buffer

    INTEGER strlen, rows, cols, i, io
    REAL, DIMENSION(:,:), ALLOCATABLE :: x

    OPEN (1, file=filename, status='old', action='read')

    !Count the number of columns

    READ(1,'(a)') buffer !read first line WITH SPACES INCLUDED
    REWIND(1) !Get back to the file beginning

    strlen = len(buffer) !Find the REAL length of a string read
    DO WHILE (buffer(strlen:strlen) == ' ')
      strlen = strlen - 1
    END DO

    cols=0 !Count the number of spaces in the first line
    DO i=0,strlen
      IF (buffer(i:i) == ' ') THEN
        cols=cols+1
      END IF
    END DO

    cols = cols+1

    !Count the number of rows

    rows = 0 !Count the number of lines in a file
    DO
      READ(1,*,iostat=io)
      IF (io/=0) EXIT
      rows = rows + 1
    END DO

    REWIND(1)

    PRINT*, 'Number of rows:', rows
    PRINT*, 'Number of columns:', cols

    ALLOCATE(x(rows,cols))

    DO i=1,rows,1
      READ(1,*) x(i,:)
!      WRITE(*,*) x(i,:)
    END DO
END FUNCTION txt_to_r_array
END MODULE matrix_txt_reader_mod