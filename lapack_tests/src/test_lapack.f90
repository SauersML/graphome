program test_all_lapack
  implicit none

  ! Parameters and variables
  integer, parameter :: n = 5
  integer :: info, i, j
  integer :: lwork, liwork
  integer :: one = 1
  integer :: kd, ldab, ldq 
  integer :: iamax_val, k
  integer, dimension(n) :: indxq
  integer, dimension(n) :: index

  real(8) :: dlamch_val, dnrm2_val, dlapy2_val, dlassq_val, dlamc3_val
  real(8) :: cs, sn, r, f, g, tau
  real(8), parameter :: alpha = 2.0d0, beta = 0.5d0

  real(8), dimension(n) :: d, e, diag, offdiag, w, z
  real(8), dimension(n, n) :: mat_a, mat_b, mat_c, q
  real(8), allocatable :: ab(:,:)

  ! Arrays and matrices
  real(8), allocatable :: a(:), b(:), c_array(:), work(:)
  integer, allocatable :: iwork(:)

  ! External functions
  real(8), external :: DLAMCH, DLANST, DLAMC3, DLAPY2, DNRM2
  integer, external :: IDAMAX

  ! External subroutines
  external :: DSBTRD, DLASET, DLARTG, DROT, DLARTV, DLARGV
  external :: DLAR2V, DSTEDC, DLASSQ, DLAED0, DCOPY, DGEMM, DLACPY
  external :: DLAMRG, DSCAL, DLAED5, DLAED6, DGEMV, DSTEQR, DLASCL
  external :: DLAEV2, DSWAP, DLASR

  ! Initialize variables
  info = 0
  lwork = 1000
  liwork = 1000
  kd = 1
  ldab = kd + 1
  ldq = n
  k = n / 2

  allocate(a(n), b(n), c_array(n))
  allocate(work(lwork))
  allocate(iwork(liwork))
  
  ! Initialize arrays
  do i = 1, n
    a(i) = 1.0d0 * i
    b(i) = 2.0d0 * i
    c_array(i) = 0.0d0
    d(i) = 1.0d0 * i
    if (i < n) e(i) = 0.1d0 * i
    w(i) = 1.0d0 * i
    z(i) = 1.0d0
    do j = 1, n
      mat_a(i,j) = i + j
      mat_b(i,j) = n - i + j
      mat_c(i,j) = 0.0d0
      q(i,j) = 0.0d0
    end do
  end do

  ! 1. DLAMCH
  dlamch_val = DLAMCH('E')
  open(unit=11, file='results/dlamch_output.txt', status='replace')
  write(11,*) 'DLAMCH("E") = ', dlamch_val
  close(11)

  ! 2. DSBTRD
  allocate(ab(ldab,n))
  ab = 0.0d0
  ab(kd+1,1:n) = d(1:n)
  if (kd > 0 .and. n > 1) ab(kd,2:n) = e(1:n-1)
  diag = 0.0d0
  offdiag = 0.0d0
  call DSBTRD('N', 'U', n, kd, ab, ldab, diag, offdiag(1:n-1), q, ldq, work, info)
  open(unit=12, file='results/dsbtrd_output.txt', status='replace')
  write(12,*) 'DSBTRD Output:'
  write(12,*) 'Diagonal:', diag
  write(12,*) 'Off-diagonal:', offdiag(1:n-1)
  write(12,*) 'Info:', info
  close(12)

  ! 3. DLASET
  call DLASET('Full', n, n, alpha, beta, mat_c, n)
  open(unit=13, file='results/dlaset_output.txt', status='replace')
  write(13,*) 'DLASET Output:'
  do i = 1, n
    write(13,'(5(F8.4,1X))') mat_c(i,:)
  end do
  close(13)
  mat_c = 0.0d0

  ! 4. DLARTG
  call DLARTG(1.0d0, 2.0d0, cs, sn, r)
  open(unit=14, file='results/dlartg_output.txt', status='replace')
  write(14,*) 'DLARTG Output:'
  write(14,*) 'CS:', cs
  write(14,*) 'SN:', sn
  write(14,*) 'R:', r
  close(14)

  ! 5. DROT
  call DROT(n, a, 1, b, 1, cs, sn)
  open(unit=15, file='results/drot_output.txt', status='replace')
  write(15,*) 'DROT Output:'
  write(15,*) 'A after rotation:', a
  write(15,*) 'B after rotation:', b
  close(15)

  ! Reinitialize a, b
  do i = 1, n
    a(i) = 1.0d0 * i
    b(i) = 2.0d0 * i
  end do

  ! 6. DLARTV
  ! Initialize arrays c_array and work (sn)
  do i = 1, n
    c_array(i) = cos(0.1d0 * i)
    work(i) = sin(0.1d0 * i)
  end do
  call DLARTV(n, a, 1, b, 1, c_array, work, 1)
  open(unit=16, file='results/dlartv_output.txt', status='replace')
  write(16,*) 'DLARTV Output:'
  write(16,*) 'A after transformation:', a
  write(16,*) 'B after transformation:', b
  close(16)

  ! Reinitialize a, b
  do i = 1, n
    a(i) = 1.0d0 * i
    b(i) = 2.0d0 * i
  end do

  ! 7. DLARGV
  call DLARGV(n, a, 1, b, 1, c_array, 1)
  open(unit=17, file='results/dlargv_output.txt', status='replace')
  write(17,*) 'DLARGV Output:'
  write(17,*) 'A after transformation:', a
  write(17,*) 'B after transformation:', b
  write(17,*) 'C (angles):', c_array
  close(17)

  ! 8. DLAR2V
  ! Reinitialize a, b, and c_array if necessary
  call DLAR2V(n, a, b, c_array, 1, c_array, work, 1)
  open(unit=18, file='results/dlar2v_output.txt', status='replace')
  write(18,*) 'DLAR2V Output:'
  write(18,*) 'A after transformation:', a
  write(18,*) 'B after transformation:', b
  write(18,*) 'C after transformation:', c_array
  close(18)

  ! 9. DSTEDC
  call DSTEDC('I', n, d, e(1:n-1), q, n, work, lwork, iwork, liwork, info)
  open(unit=19, file='results/dstedc_output.txt', status='replace')
  write(19,*) 'DSTEDC Output:'
  write(19,*) 'Eigenvalues:', d
  write(19,*) 'Eigenvectors (columns):'
  do i = 1, n
    write(19,'(5(F8.4,1X))') q(1:n,i)
  end do
  write(19,*) 'Info:', info
  close(19)

  ! 10. DLANST
  dlassq_val = DLANST('M', n, d, e(1:n-1))
  open(unit=20, file='results/dlanst_output.txt', status='replace')
  write(20,*) 'DLANST Output:'
  write(20,*) 'Max norm:', dlassq_val
  close(20)

  ! 11. DLASSQ
  f = 0.0d0
  g = 1.0d0
  call DLASSQ(n, a, 1, f, g)
  open(unit=21, file='results/dlassq_output.txt', status='replace')
  write(21,*) 'DLASSQ Output:'
  write(21,*) 'Scale (F):', f
  write(21,*) 'Sum of squares (G):', g
  close(21)

  ! Reinitialize a
  do i = 1, n
    a(i) = 1.0d0 * i
  end do

  ! 12. DLAED0
  ! Reallocate work and iwork arrays with sufficient size
  deallocate(work)
  deallocate(iwork)
  lwork = 4*n + n*n
  liwork = 6 + 6*n + 5*n*int(log(real(n))/log(2.0d0))
  allocate(work(lwork))
  allocate(iwork(liwork))
  call DLAED0(n, n, d, e(1:n-1), q, n, work, iwork, info)
  open(unit=22, file='results/dlaed0_output.txt', status='replace')
  write(22,*) 'DLAED0 Output:'
  write(22,*) 'Eigenvalues:', d
  write(22,*) 'Eigenvectors (columns):'
  do i = 1, n
    write(22,'(5(F8.4,1X))') q(1:n,i)
  end do
  write(22,*) 'Info:', info
  close(22)

  ! 13. DCOPY
  call DCOPY(n, a, 1, b, 1)
  open(unit=23, file='results/dcopy_output.txt', status='replace')
  write(23,*) 'DCOPY Output:'
  write(23,*) 'Copied array B:', b
  close(23)

  ! 14. DGEMM
  call DGEMM('N', 'N', n, n, n, alpha, mat_a, n, mat_b, n, beta, mat_c, n)
  open(unit=24, file='results/dgemm_output.txt', status='replace')
  write(24,*) 'DGEMM Output:'
  write(24,*) 'Resultant Matrix C:'
  do i = 1, n
    write(24,'(5(F8.4,1X))') mat_c(i,:)
  end do
  close(24)
  mat_c = 0.0d0

  ! 15. DLACPY
  call DLACPY('A', n, n, mat_a, n, mat_c, n)
  open(unit=25, file='results/dlacpy_output.txt', status='replace')
  write(25,*) 'DLACPY Output:'
  write(25,*) 'Copied Matrix C:'
  do i = 1, n
    write(25,'(5(F8.4,1X))') mat_c(i,:)
  end do
  close(25)
  mat_c = 0.0d0

  ! 16. DLAMRG
  ! Initialize index array
  do i = 1, n/2
    a(i) = 1.0d0 * (2*i - 1)
  end do
  do i = n/2+1, n
    a(i) = 1.0d0 * (2*(i - n/2))
  end do
  call DLAMRG(n/2, n - n/2, a, 1, 1, index)
  open(unit=29, file='results/dlamrg_output.txt', status='replace')
  write(29,*) 'DLAMRG Output:'
  write(29,*) 'Merged indices:', index
  close(29)

  ! 17. DSCAL
  call DSCAL(n, 2.0d0, a, 1)
  open(unit=30, file='results/dscal_output.txt', status='replace')
  write(30,*) 'DSCAL Output:'
  write(30,*) 'Scaled array A:', a
  close(30)
  ! Reinitialize a
  do i = 1, n
    a(i) = 1.0d0 * i
  end do

  ! 18. DLAMC3
  dlamc3_val = DLAMC3(1.0d0, 2.0d0)
  open(unit=32, file='results/dlamc3_output.txt', status='replace')
  write(32,*) 'DLAMC3 Output:'
  write(32,*) 'Result:', dlamc3_val
  close(32)

  ! 19. DNRM2
  dnrm2_val = DNRM2(n, a, 1)
  open(unit=33, file='results/dnrm2_output.txt', status='replace')
  write(33,*) 'DNRM2 Output:'
  write(33,*) 'Euclidean norm of A:', dnrm2_val
  close(33)

  ! 20. DLAED5
  call DLAED5(1, d(1:2), z(1:2), c_array(1:2), 0.0d0, f)
  open(unit=34, file='results/dlaed5_output.txt', status='replace')
  write(34,*) 'DLAED5 Output:'
  write(34,*) 'Computed eigenvalue (lambda):', f
  write(34,*) 'Updated delta:', c_array(1:2)
  close(34)

  ! 21. DLAED6
  ! Initialize variables
  tau = 0.0d0
  f = 0.0d0
  call DLAED6(1, .true., 0.0d0, d(1:3), z(1:3), f, tau, info)
  open(unit=35, file='results/dlaed6_output.txt', status='replace')
  write(35,*) 'DLAED6 Output:'
  write(35,*) 'Computed Tau:', tau
  write(35,*) 'Info:', info
  close(35)

  ! 22. DGEMV
  call DGEMV('N', n, n, alpha, mat_a, n, a, 1, beta, b, 1)
  open(unit=36, file='results/dgemv_output.txt', status='replace')
  write(36,*) 'DGEMV Output:'
  write(36,*) 'Resultant B:', b
  close(36)

  ! Reinitialize a, b
  do i = 1, n
    a(i) = 1.0d0 * i
    b(i) = 2.0d0 * i
  end do

  ! 23. DSTEQR
  call DSTEQR('I', n, d, e(1:n-1), q, n, work, info)
  open(unit=37, file='results/dsteqr_output.txt', status='replace')
  write(37,*) 'DSTEQR Output:'
  write(37,*) 'Eigenvalues:', d
  write(37,*) 'Eigenvectors (columns):'
  do i = 1, n
    write(37,'(5(F8.4,1X))') q(1:n,i)
  end do
  write(37,*) 'Info:', info
  close(37)

  ! 24. DLASCL
  call DLASCL('G', 0, 0, 1.0d0, 2.0d0, n, n, mat_a, n, info)
  open(unit=38, file='results/dlascl_output.txt', status='replace')
  write(38,*) 'DLASCL Output:'
  write(38,*) 'Scaled Matrix A:'
  do i = 1, n
    write(38,'(5(F8.4,1X))') mat_a(i,:)
  end do
  write(38,*) 'Info:', info
  close(38)

  ! Reinitialize mat_a
  do i = 1, n
    do j = 1, n
      mat_a(i,j) = i + j
    end do
  end do

  ! 25. DLAEV2
  call DLAEV2(1.0d0, 2.0d0, 3.0d0, f, g, cs, sn)
  open(unit=39, file='results/dlaev2_output.txt', status='replace')
  write(39,*) 'DLAEV2 Output:'
  write(39,*) 'Eigenvalues:', f, g
  write(39,*) 'CS:', cs
  write(39,*) 'SN:', sn
  close(39)

  ! 26. DLASR
  ! Initialize c_array and work to contain rotation parameters
  do i = 1, n
    c_array(i) = cos(0.1d0 * i)
    work(i) = sin(0.1d0 * i)
  end do
  call DLASR('L', 'V', 'F', n, n, c_array, work, mat_a, n)
  open(unit=40, file='results/dlasr_output.txt', status='replace')
  write(40,*) 'DLASR Output:'
  write(40,*) 'Matrix A after DLASR:'
  do i = 1, n
    write(40,'(5(F8.4,1X))') mat_a(i,:)
  end do
  close(40)

  ! 27. DSWAP
  call DSWAP(n, a, 1, b, 1)
  open(unit=41, file='results/dswap_output.txt', status='replace')
  write(41,*) 'DSWAP Output:'
  write(41,*) 'A after DSWAP:', a
  write(41,*) 'B after DSWAP:', b
  close(41)

  ! Deallocate
  deallocate(a, b, c_array, work, iwork, ab)

end program test_all_lapack
