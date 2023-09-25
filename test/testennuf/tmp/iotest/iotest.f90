! Created by  on 22/09/2023.

program iotest
    real(kind=4) :: arr(3,2,5)
    integer :: unit, reclen
    unit = 33

    ! Read array
    OPEN(unit, FILE="itest.dat",&
     FORM="UNFORMATTED", STATUS="UNKNOWN", ACTION="READ", ACCESS='STREAM')
    READ(unit) arr
    CLOSE(unit)
    print*, arr

    ! Write array
    INQUIRE(IOLENGTH = reclen)arr
    OPEN(unit, file='otest.dat', status='replace', action='write', form='unformatted', access='stream')
    WRITE(unit) arr
    CLOSE(unit)

end program iotest