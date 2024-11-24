program test_all_lapack
  implicit none

  ! Parameters and variables
  integer, parameter :: n = 5
  integer :: info, i, j
  integer :: lwork, liwork
  integer :: one = 1
  integer :: kd, ldab, ldq 
  integer :: iamax_val, k
  integer, dimension(n) :: indxq, index

  real(8) :: dlamch_val, dnrm2_val, dlapy2_val, dlassq_val, dlamc3_val
  real(8) :: cs, sn, r, f, g, tau
  real(8), parameter :: alpha = 2.0d0, beta = 0.5d0

  real(8), dimension(n) :: d, e, diag, offdiag, w, z
  real(8), dimension(n, n) :: mat_a, mat_b, mat_c, q
  real(8), dimension(ldab, n) :: ab

  ! Arrays and matrices
  real(8), allocatable :: a(:), b(:), c(:), work(:), work_alloc(:)
  integer, allocatable :: iwork(:), iwork_alloc(:)

  ! External functions
  real(8), external :: DLAMCH, DLANST, DLAMC3, DLAPY2, DNRM2
  integer, external :: IDAMAX

  ! External subroutines
  external :: DSBTRD, DLASET, DLARTG, DROT, DLARTV, DLARGV
  external :: DLAR2V, DSTEDC, DLASSQ, DLAED0, DCOPY, DGEMM, DLACPY
  external :: DLAED1, DLAED2, DLAMRG, DSCAL, DLAED3, DLAED4, DLAED5
  external :: DLAED6, DLAED7, DLAED8, DLAED9, DLAEDA, DGEMV
  external :: DSTEQR, DLASCL, DLAEV2, DSWAP, DLASR

  ! Initialize variables
  info = 0
  lwork = 1000
  liwork = 1000
  kd = 1
  ldab = kd + 1
  ldq = n
  k = n / 2

  allocate(a(n), b(n), c(n))
  allocate(work(lwork))
  allocate(iwork(liwork))
  allocate(work_alloc(lwork))
  allocate(iwork_alloc(liwork))

  ! Initialize arrays
  do i = 1, n
    a(i) = 1.0d0 * i
    b(i) = 2.0d0 * i
    c(i) = 0.0d0
    d(i) = 1.0d0 * i
    e(i) = 0.1d0 * i
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
  ab = 0.0d0
  ab(kd+1,:) = d(1:n)
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
  call DLARTV(n, a, 1, b, 1, cs, sn, 1)
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
  call DLARGV(n, a, 1, b, 1, c, 1)
  open(unit=17, file='results/dlargv_output.txt', status='replace')
  write(17,*) 'DLARGV Output:'
  write(17,*) 'A after transformation:', a
  write(17,*) 'B after transformation:', b
  write(17,*) 'C (angles):', c
  close(17)

  ! Reinitialize a, b, c
  do i = 1, n
    a(i) = 1.0d0 * i
    b(i) = 2.0d0 * i
    c(i) = 0.0d0
  end do

  ! 8. DLAR2V
  call DLAR2V(n, a, b, c, 1, cs, sn, 1)
  open(unit=18, file='results/dlar2v_output.txt', status='replace')
  write(18,*) 'DLAR2V Output:'
  write(18,*) 'A after transformation:', a
  write(18,*) 'B after transformation:', b
  write(18,*) 'C after transformation:', c
  close(18)
  ! Reinitialize as needed

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
  ! Reinitialize d, e, q

  ! 10. DLANST
  dlassq_val = DLANST('M', n, d, e(1:n-1))
  open(unit=20, file='results/dlanst_output.txt', status='replace')
  write(20,*) 'DLANST Output:'
  write(20,*) 'Max abs sum:', dlassq_val
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
  allocate(work_alloc(4*n + n**2))
  allocate(iwork_alloc(2*n))
  call DLAED0(1, n, n, d, e(1:n-1), q, n, work_alloc, iwork_alloc, info)
  open(unit=22, file='results/dlaed0_output.txt', status='replace')
  write(22,*) 'DLAED0 Output:'
  write(22,*) 'Eigenvalues:', d
  write(22,*) 'Eigenvectors (columns):'
  do i = 1, n
    write(22,'(5(F8.4,1X))') q(1:n,i)
  end do
  write(22,*) 'Info:', info
  close(22)
  deallocate(work_alloc)
  deallocate(iwork_alloc)
  ! Reinitialize d, e, q

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

  ! 16. DLAED1
  allocate(work_alloc(4*n))
  allocate(iwork_alloc(4*n))
  do i = 1, n
    indxq(i) = i
  end do
  call DLAED1(n, d, q, n, indxq, 0.0d0, n/2, work_alloc, iwork_alloc, info)
  open(unit=26, file='results/dlaed1_output.txt', status='replace')
  write(26,*) 'DLAED1 Output:'
  write(26,*) 'Updated Eigenvalues:', d
  write(26,*) 'Updated Eigenvectors:'
  do i = 1, n
    write(26,'(5(F8.4,1X))') q(1:n,i)
  end do
  write(26,*) 'Info:', info
  close(26)
  deallocate(work_alloc)
  deallocate(iwork_alloc)

  ! 17. DLAED2
  ! DLAED2 is internal to DLAED1 and not intended for separate use.
  ! Skipping direct call to DLAED2.

  ! 18. IDAMAX
  iamax_val = IDAMAX(n, a, 1)
  open(unit=27, file='results/idamax_output.txt', status='replace')
  write(27,*) 'IDAMAX Output:'
  write(27,*) 'Index of max element in A:', iamax_val
  close(27)

  ! 19. DLAPY2
  dlapy2_val = DLAPY2(3.0d0, 4.0d0)
  open(unit=28, file='results/dlapy2_output.txt', status='replace')
  write(28,*) 'DLAPY2 Output:'
  write(28,*) 'Result (should be 5):', dlapy2_val
  close(28)

  ! 20. DLAMRG
  integer, dimension(n) :: index
  call DLAMRG(n/2, n - n/2, a, 1, 1, index)
  open(unit=29, file='results/dlamrg_output.txt', status='replace')
  write(29,*) 'DLAMRG Output:'
  write(29,*) 'Merged indices:', index
  close(29)

  ! 21. DSCAL
  call DSCAL(n, 2.0d0, a, 1)
  open(unit=30, file='results/dscal_output.txt', status='replace')
  write(30,*) 'DSCAL Output:'
  write(30,*) 'Scaled array A:', a
  close(30)
  ! Reinitialize a
  do i = 1, n
    a(i) = 1.0d0 * i
  end do

  ! 22. DLAED3
  allocate(work_alloc(4*n))
  allocate(iwork_alloc(4*n))
  call DLAED3(n, k, d, q, n, 0.0d0, c, c, indxq, iwork_alloc, work_alloc, info)
  open(unit=31, file='results/dlaed3_output.txt', status='replace')
  write(31,*) 'DLAED3 Output:'
  write(31,*) 'Updated Eigenvalues:', d
  write(31,*) 'Updated Eigenvectors:'
  do i = 1, n
    write(31,'(5(F8.4,1X))') q(1:n,i)
  end do
  write(31,*) 'Info:', info
  close(31)
  deallocate(work_alloc)
  deallocate(iwork_alloc)

  ! 23. DLAMC3
  dlamc3_val = DLAMC3(1.0d0, 2.0d0)
  open(unit=32, file='results/dlamc3_output.txt', status='replace')
  write(32,*) 'DLAMC3 Output:'
  write(32,*) 'Result:', dlamc3_val
  close(32)

  ! 24. DNRM2
  dnrm2_val = DNRM2(n, a, 1)
  open(unit=33, file='results/dnrm2_output.txt', status='replace')
  write(33,*) 'DNRM2 Output:'
  write(33,*) 'Euclidean norm of A:', dnrm2_val
  close(33)

  ! 25. DLAED4
  ! DLAED4 computes one eigenvalue, but needs inputs from DLAED3/DLAED2
  ! We'll skip calling DLAED4 directly as it's used internally.

  ! 26. DLAED5
  call DLAED5(1, d(1:2), z(1:2), c(1:2), 0.0d0, f)
  open(unit=34, file='results/dlaed5_output.txt', status='replace')
  write(34,*) 'DLAED5 Output:'
  write(34,*) 'Computed eigenvalue (lambda):', f
  write(34,*) 'Updated c:', c(1:2)
  close(34)

  ! 27. DLAED6
  call DLAED6(1, .true., 0.0d0, d(1:3), z(1:3), f, tau, info)
  open(unit=35, file='results/dlaed6_output.txt', status='replace')
  write(35,*) 'DLAED6 Output:'
  write(35,*) 'Computed Tau:', tau
  write(35,*) 'Info:', info
  close(35)

  ! 28. DLAED7
  ! As the input data for DLAED7 is extensive and not straightforward,
  ! we'll note that DLAED7 is usually called within higher-level routines.
  ! For this test, we'll skip calling DLAED7 directly.

  ! 29. DLAED8
  ! DLAED8 is also an internal routine called from DLAED7.
  ! We'll skip direct call.

  ! 30. DLAED9
  ! DLAED9 computes updated eigenvalues and eigenvectors.
  ! It's used within DLAED7/DLAED8.
  ! We'll skip direct call.

  ! 31. DLAEDA
  ! DLAEDA prepares the Z vector within DLAED7.
  ! Skipping direct call.

  ! 32. DGEMV
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

  ! 33. DSTEQR
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
  ! Reinitialize d, e, q

  ! 34. DLASCL
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

  ! 35. DLAEV2
  call DLAEV2(1.0d0, 2.0d0, 3.0d0, f, g, cs, sn)
  open(unit=39, file='results/dlaev2_output.txt', status='replace')
  write(39,*) 'DLAEV2 Output:'
  write(39,*) 'Eigenvalues:', f, g
  write(39,*) 'CS:', cs
  write(39,*) 'SN:', sn
  close(39)

  ! 36. DLASR
  call DLASR('L', 'V', 'F', n, n, c, c, mat_a, n)
  open(unit=40, file='results/dlasr_output.txt', status='replace')
  write(40,*) 'DLASR Output:'
  write(40,*) 'Matrix A after DLASR:'
  do i = 1, n
    write(40,'(5(F8.4,1X))') mat_a(i,:)
  end do
  close(40)
  ! Reinitialize mat_a

  ! 37. DSWAP
  call DSWAP(n, a, 1, b, 1)
  open(unit=41, file='results/dswap_output.txt', status='replace')
  write(41,*) 'DSWAP Output:'
  write(41,*) 'A after DSWAP:', a
  write(41,*) 'B after DSWAP:', b
  close(41)

  ! Deallocate
  deallocate(a, b, c, work, iwork, work_alloc, iwork_alloc)

end program test_all_lapack
