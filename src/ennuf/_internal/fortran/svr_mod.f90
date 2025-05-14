PROGRAM svr_program
IMPLICIT NONE

! INTEGER, PARAMETER :: precision=4
! REAL(kind = precision), DIMENSION(2)  :: y_out
! CALL svr(3,2,2,y_out, 4)

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
      ! CALL print_matrix_int("support vectors",supp_vectors)
      ! CALL print_vector_int
      ! CALL print_matrix_int("support vectors",supp_vectors-x_in)
      write(*,*) "input"
      write(*,*) x_in

      temp=SPREAD(x_in,1,n_supp_vectors)
      write (*,*) "subtracting the two"
      ! write (*,*) supp_vectors(3,:)
      ! write (*,*) temp(3,:)
      write(*,*) supp_vectors-temp

      y_out = exp(-1*NORM2(supp_vectors-temp,2)**2)
      write(*,*) y_out
   END SUBROUTINE rbf   


   SUBROUTINE svr(x_in, supp_vectors, dual_coef, n_supp_vectors, n_dims, y_out, intercept)
      IMPLICIT NONE

      INTEGER, PARAMETER                                    :: precision = 4
      INTEGER :: j
      INTEGER, INTENT(in) :: n_supp_vectors, n_dims, intercept
      ! real :: supp_vectors(n_supp_vectors,n_dims), dual_coef(n_supp_vectors), x_in(n_in, n_dims), y_out(n_in), temp(n_dims)
      REAL(kind = precision), DIMENSION(n_supp_vectors,n_dims), INTENT(in) :: supp_vectors
      REAL(kind = precision), DIMENSION(n_supp_vectors), INTENT(in) :: dual_coef
      REAL(kind = precision), DIMENSION(1,n_dims), INTENT(in)  :: x_in
      REAL(kind = precision), DIMENSION(n_supp_vectors)  :: temp
      REAL(kind = precision), DIMENSION(1),  INTENT(out)  :: y_out
      
      write(*,*) precision
      ! supp_vectors = reshape([1, 2, 3, 4, 5, 6], [n_supp_vectors, n_dims], ORDER=[2,1])
      write(*,*) supp_vectors
      ! x_in = reshape([1, 2, 3, 4], [n_in, n_dims], ORDER=[2,1])
      ! dual_coef = [2,4,6]
      write(*,*) size(x_in,1)
      do j=1, size(x_in,1)
         ! write(*,*) j
         ! write(*,*) x_in(j,:)
         CALL rbf(n_supp_vectors, n_dims, supp_vectors, x_in(j,:), temp)
         y_out(j) = DOT_PRODUCT(dual_coef, temp)+intercept
         write(*,*) y_out(j)
      end do
      
   END SUBROUTINE svr

! CONVENIENCE ROUTINES TO PRINT IN ROW-COLUMN ORDER
subroutine print_vector_int(title,arr)
character(len=*),intent(in)  :: title
integer,intent(in)           :: arr(:)
   call print_matrix_int(title,reshape(arr,[1,shape(arr)]))
end subroutine print_vector_int

subroutine print_matrix_int(title,arr)
!@(#) print small 2d integer arrays in row-column format
character(len=*),parameter :: all='(" > ",*(g0,1x))' ! a handy format
character(len=*),intent(in)  :: title
integer,intent(in)           :: arr(:,:)
integer                      :: i
character(len=:),allocatable :: biggest

   print all
   print all, trim(title)
   biggest='           '  ! make buffer to write integer into
   ! find how many characters to use for integers
   write(biggest,'(i0)')ceiling(log10(real(maxval(abs(arr)))))+2
   ! use this format to write a row
   biggest='(" > [",*(i'//trim(biggest)//':,","))'
   ! print one row of array at a time
   do i=1,size(arr,dim=1)
      write(*,fmt=biggest,advance='no')arr(i,:)
      write(*,'(" ]")')
   enddo

end subroutine print_matrix_int

END PROGRAM