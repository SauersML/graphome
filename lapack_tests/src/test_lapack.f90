! src/test_lapack.f90
program test_all_lapack
  implicit none

  ! Variable declarations

  ! Integers
  integer :: i, info
  integer :: ldz, n, n_e, n_z, one
  integer :: iamax_val
  integer :: n_rows, n_cols
  integer :: k, nl, nr, icompq, lwork, m
  integer :: unit_number
  integer :: nsplit, neig

  ! Reals
  real(8) :: dlamch_val, dnrm2_val, dlapy2_val, dlassq_val
  real(8) :: cs, sn, f, g, tau, r
  real(8), parameter :: alpha = 2.0d0, beta = 0.5d0
  real(8) :: temp1, temp2

  ! Arrays
  real(8), dimension(3) :: a, b, c, work
  real(8), dimension(3,3) :: mat_a, mat_b, mat_c
  real(8), dimension(5) :: e, diag, offdiag, work5
  real(8), dimension(5) :: d
  real(8), dimension(4) :: z
  real(8), allocatable :: work_alloc(:)
  integer, allocatable :: iwork_alloc(:)
  integer, dimension(5) :: iwork
  real(8), dimension(3,2) :: mat_3x2
  real(8), dimension(2,3) :: mat_2x3

  ! File handling
  logical :: bwed
  character(len=20) :: filename

  ! Declare external functions
  real(8), external :: DLAMCH, DLANST, DLAPY2, DLAMC3, DNRM2
  integer, external :: IDAMAX

  ! External subroutines (BLAS and LAPACK routines)
  external :: DLASR, DSBTRD, DLASET, DLARTG, DROT, DLARTV, DLARGV
  external :: DLAR2V, DSTEDC, DLASSQ, DLAED0, DCOPY, DGEMM, DLACPY
  external :: DLAED1, DLAED2, DLAMRG, DSCAL, DLAED3, DLAED4, DLAED5
  external :: DLAED6, DLAED7, DLAED8, DLAED9, DLAEDA, DGEMV
  external :: DSTEQR, DLASCL, DLAEV2, DSWAP

  ! Initialize variables
  ldz = 3
  n = 3
  n_e = 5
  n_z = 4
  one = 1

  a = [1.0d0, 2.0d0, 3.0d0]
  b = [4.0d0, 5.0d0, 6.0d0]
  c = [0.0d0, 0.0d0, 0.0d0]
  work = [0.0d0, 0.0d0, 0.0d0]

  mat_a = reshape([1.0d0, 2.0d0, 3.0d0, &
                   4.0d0, 5.0d0, 6.0d0, &
                   7.0d0, 8.0d0, 9.0d0], [3,3])

  mat_b = reshape([9.0d0, 8.0d0, 7.0d0, &
                   6.0d0, 5.0d0, 4.0d0, &
                   3.0d0, 2.0d0, 1.0d0], [3,3])

  mat_c = 0.0d0

  e = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0]
  diag = [5.0d0, 4.0d0, 3.0d0, 2.0d0, 1.0d0]
  offdiag = [1.0d0, 1.0d0, 1.0d0, 1.0d0]
  work5 = [0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0]
  d = [0.0d0, 0.0d0, 0.0d0, 0.0d0, 0.0d0]

  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0]

  mat_3x2 = reshape([1.0d0, 4.0d0, 2.0d0, &
                     5.0d0, 3.0d0, 6.0d0], [3,2])

  mat_2x3 = reshape([1.0d0, 2.0d0, 3.0d0, &
                     4.0d0, 5.0d0, 6.0d0], [2,3])

  iwork = [0,0,0,0,0]

  ! Ensure the "results" directory exists
  inquire(file='results', exist=bwed)
  if (.not.bwed) then
     inquire(file='.', exist=bwed)
     if (bwed) then
        ! Create the directory
        call system('mkdir -p results')
     else
        print *, 'Cannot find the current directory.'
        stop
     end if
  end if

  ! 1. DLAMCH
  dlamch_val = DLAMCH('E')
  open(unit=10, file='results/dlamch_output.txt', status='replace')
  write(10,*) 'DLAMCH("E") = ', dlamch_val
  close(10)

  ! 2. DSBTRD
  ! For DSBTRD, create a simple symmetric band matrix
  integer, parameter :: kd = 1
  real(8), dimension(kd+1,n) :: AB
  real(8), dimension(n) :: work_dsbt
  AB = 0.0d0
  AB(kd+1,:) = [4.0d0, 5.0d0, 6.0d0]
  AB(kd,2:n) = [1.0d0, 2.0d0]
  diag(1:n) = 0.0d0
  offdiag(1:n-1) = 0.0d0
  work_dsbt(1:n) = 0.0d0
  call DSBTRD('N', 'U', n, kd, AB, kd+1, diag(1:n), offdiag(1:n-1), mat_c, n, work_dsbt, info)
  open(unit=10, file='results/dsbtrd_output.txt', status='replace')
    write(10,*) 'DSBTRD Output:'
    write(10,*) 'Diagonal:', diag(1:n)
    write(10,*) 'Off-diagonal:', offdiag(1:n-1)
    write(10,*) 'Info:', info
  close(10)

  ! 3. DLASET
  call DLASET('U', n, n, alpha, beta, mat_c, n)
  open(unit=10, file='results/dlaset_output.txt', status='replace')
    write(10,*) 'DLASET Output:'
    do i = 1, n
      write(10, '(3(F8.4,1X))') mat_c(i,:)
    end do
  close(10)
  mat_c = 0.0d0

  ! 4. DLARTG
  call DLARTG(1.0d0, 2.0d0, cs, sn, r)
  open(unit=10, file='results/dlartg_output.txt', status='replace')
    write(10,*) 'DLARTG Output:'
    write(10,*) 'CS:', cs
    write(10,*) 'SN:', sn
    write(10,*) 'R:', r
  close(10)

  ! 5. DROT
  call DROT(n, a, 1, b, 1, cs, sn)
  open(unit=10, file='results/drot_output.txt', status='replace')
    write(10,*) 'DROT Output:'
    write(10,*) 'A after rotation:', a
    write(10,*) 'B after rotation:', b
  close(10)
  a = [1.0d0, 2.0d0, 3.0d0]
  b = [4.0d0, 5.0d0, 6.0d0]

  ! 6. DLARTV
  call DLARTV(n, a, 1, b, 1, cs, sn, 1)
  open(unit=10, file='results/dlartv_output.txt', status='replace')
    write(10,*) 'DLARTV Output:'
    write(10,*) 'A after DLARTV:', a
    write(10,*) 'B after DLARTV:', b
  close(10)
  a = [1.0d0, 2.0d0, 3.0d0]
  b = [4.0d0, 5.0d0, 6.0d0]

  ! 7. DLARGV
  call DLARGV(n, a, 1, b, 1, work, 1)
  open(unit=10, file='results/dlargv_output.txt', status='replace')
    write(10,*) 'DLARGV Output:'
    write(10,*) 'A after DLARGV:', a
    write(10,*) 'B after DLARGV:', b
    write(10,*) 'Work (angles):', work
  close(10)
  a = [1.0d0, 2.0d0, 3.0d0]
  b = [4.0d0, 5.0d0, 6.0d0]

  ! 8. DLAR2V
  call DLAR2V(2, a(1), b(1), c(1), 1, cs, sn, 1)
  open(unit=10, file='results/dlar2v_output.txt', status='replace')
    write(10,*) 'DLAR2V Output:'
    write(10,*) 'A:', a(1:2)
    write(10,*) 'B:', b(1:2)
    write(10,*) 'C:', c(1:2)
  close(10)
  a = [1.0d0, 2.0d0, 3.0d0]
  b = [4.0d0, 5.0d0, 6.0d0]
  c = [7.0d0, 8.0d0, 9.0d0]

  ! 9. DSTEDC
  icompq = 1
  lwork = 1 + 3*n_e + 2*n_e**2
  allocate(work_alloc(lwork))
  allocate(iwork_alloc(3*n_e))
  diag = [5.0d0, 4.0d0, 3.0d0, 2.0d0, 1.0d0]
  e = [1.0d0, 1.0d0, 1.0d0, 1.0d0]
  call DSTEDC('I', n_e, diag, e, mat_a, n_e, work_alloc, lwork, iwork_alloc, 3*n_e, info)
  open(unit=10, file='results/dstedc_output.txt', status='replace')
    write(10,*) 'DSTEDC Output:'
    write(10,*) 'Eigenvalues:', diag
    write(10,*) 'Eigenvectors (columns):'
    do i = 1, n_e
      write(10, '(5(F8.4,1X))') mat_a(1:n_e,i)
    end do
    write(10,*) 'Info:', info
  close(10)
  deallocate(work_alloc)
  deallocate(iwork_alloc)
  diag = [5.0d0, 4.0d0, 3.0d0, 2.0d0, 1.0d0]
  e = [1.0d0, 1.0d0, 1.0d0, 1.0d0]

  ! 10. DLANST
  dlassq_val = DLANST('M', n_e, diag, e)
  open(unit=10, file='results/dlanst_output.txt', status='replace')
    write(10,*) 'DLANST Output:'
    write(10,*) 'Max(abs(diag(i)) + abs(e(i))) =', dlassq_val
  close(10)

  ! 11. DLASSQ
  f = 0.0d0
  g = 1.0d0
  call DLASSQ(n, a, one, f, g)
  open(unit=10, file='results/dlassq_output.txt', status='replace')
    write(10,*) 'DLASSQ Output:'
    write(10,*) 'Scale (F):', f
    write(10,*) 'Sum of squares (G):', g
  close(10)

  ! 12. DLAED0
  allocate(work_alloc(4*n_e + n_e**2))
  allocate(iwork_alloc(2*n_e))
  diag = [5.0d0, 4.0d0, 3.0d0, 2.0d0, 1.0d0]
  e = [1.0d0, 1.0d0, 1.0d0, 1.0d0]
  call DLAED0(n_e, diag, e, mat_a, n_e, work_alloc, iwork_alloc, info)
  open(unit=10, file='results/dlaed0_output.txt', status='replace')
    write(10,*) 'DLAED0 Output:'
    write(10,*) 'Eigenvalues:', diag
    write(10,*) 'Eigenvectors:'
    do i = 1, n_e
      write(10, '(5(F8.4,1X))') mat_a(1:n_e,i)
    end do
    write(10,*) 'Info:', info
  close(10)
  deallocate(work_alloc)
  deallocate(iwork_alloc)

  ! 13. DCOPY
  call DCOPY(n, a, 1, c, 1)
  open(unit=10, file='results/dcopy_output.txt', status='replace')
    write(10,*) 'DCOPY Output:'
    write(10,*) 'Copied array C:', c
  close(10)
  c = [0.0d0, 0.0d0, 0.0d0]

  ! 14. DGEMM
  call DGEMM('N', 'N', 3, 3, 2, alpha, mat_3x2, 3, mat_2x3, 2, beta, mat_c, 3)
  open(unit=10, file='results/dgemm_output.txt', status='replace')
    write(10,*) 'DGEMM Output:'
    write(10,*) 'Resulting Matrix C:'
    do i = 1, 3
      write(10, '(3(F8.4,1X))') mat_c(i,1:3)
    end do
  close(10)
  mat_c = 0.0d0

  ! 15. DLACPY
  call DLACPY('A', 3, 3, mat_a, 3, mat_c, 3)
  open(unit=10, file='results/dlacpy_output.txt', status='replace')
    write(10,*) 'DLACPY Output:'
    write(10,*) 'Copied Matrix C:'
    do i = 1, 3
      write(10, '(3(F8.4,1X))') mat_c(i,:)
    end do
  close(10)
  mat_c = 0.0d0

  ! 16. DLAED1
  allocate(work_alloc(4*n_e))
  allocate(iwork_alloc(2*n_e))
  k = 0
  diag = [5.0d0, 4.0d0, 3.0d0, 2.0d0, 1.0d0]
  e = [1.0d0, 1.0d0, 1.0d0, 1.0d0]
  z(1:n_e) = 1.0d0
  call DLAED1(n_e, diag, e, z(1:n_e), mat_a, n_e, work_alloc, iwork_alloc, info)
  open(unit=10, file='results/dlaed1_output.txt', status='replace')
    write(10,*) 'DLAED1 Output:'
    write(10,*) 'Updated Eigenvalues:', diag
    write(10,*) 'Updated Eigenvectors:'
    do i = 1, n_e
      write(10, '(5(F8.4,1X))') mat_a(1:n_e,i)
    end do
    write(10,*) 'Info:', info
  close(10)
  deallocate(work_alloc)
  deallocate(iwork_alloc)

  ! 17. DLAED2
  ! DLAED2 is called inside DLAED1, and is not intended for standalone use.
  ! For completeness, we note its presence but cannot call it directly.

  ! 18. IDAMAX
  iamax_val = IDAMAX(n, a, 1)
  open(unit=10, file='results/idamax_output.txt', status='replace')
    write(10,*) 'IDAMAX Output:'
    write(10,*) 'Index of max element in A:', iamax_val
  close(10)

  ! 19. DLAPY2
  dlapy2_val = DLAPY2(3.0d0, 4.0d0)
  open(unit=10, file='results/dlapy2_output.txt', status='replace')
    write(10,*) 'DLAPY2 Output:'
    write(10,*) 'Result (should be 5):', dlapy2_val
  close(10)

  ! 20. DLAMRG
  nl = 2
  nr = 3
  allocate(iwork_alloc(nl+nr))
  diag(1) = 1.0d0
  diag(2) = 3.0d0
  diag(3) = 2.0d0
  diag(4) = 4.0d0
  diag(5) = 5.0d0
  call DLAMRG(nl, nr, diag, 1, 1, iwork_alloc)
  open(unit=10, file='results/dlamrg_output.txt', status='replace')
    write(10,*) 'DLAMRG Output:'
    write(10,*) 'Merged indices:', iwork_alloc(1:(nl+nr))
  close(10)
  deallocate(iwork_alloc)

  ! 21. DSCAL
  call DSCAL(n, 2.0d0, a, 1)
  open(unit=10, file='results/dscal_output.txt', status='replace')
    write(10,*) 'DSCAL Output:'
    write(10,*) 'Scaled A:', a
  close(10)
  a = [1.0d0, 2.0d0, 3.0d0]

  ! 22. DLAED3
  allocate(work_alloc(4*n_e))
  allocate(iwork_alloc(2*n_e))
  diag = [5.0d0, 4.0d0, 3.0d0, 2.0d0, 1.0d0]
  z(1:n_e) = 1.0d0
  k = n_e
  call DLAED3(n_e, k, diag, z(1:n_e), mat_a, n_e, work_alloc, iwork_alloc, info)
  open(unit=10, file='results/dlaed3_output.txt', status='replace')
    write(10,*) 'DLAED3 Output:'
    write(10,*) 'Updated Eigenvalues:', diag
    write(10,*) 'Updated Eigenvectors:'
    do i = 1, n_e
      write(10, '(5(F8.4,1X))') mat_a(1:n_e,i)
    end do
    write(10,*) 'Info:', info
  close(10)
  deallocate(work_alloc)
  deallocate(iwork_alloc)

  ! 23. DLAMC3
  temp1 = 1.0d0
  temp2 = 2.0d0
  dlamch_val = DLAMC3(temp1, temp2)
  open(unit=10, file='results/dlamc3_output.txt', status='replace')
    write(10,*) 'DLAMC3 Output:'
    write(10,*) 'Result:', dlamch_val
  close(10)

  ! 24. DNRM2
  dnrm2_val = DNRM2(n, a, 1)
  open(unit=10, file='results/dnrm2_output.txt', status='replace')
    write(10,*) 'DNRM2 Output:'
    write(10,*) 'Euclidean norm of A:', dnrm2_val
  close(10)

  ! 25. DLAED4
  call DLAED4(n_e, 1, diag, e, z(1:n_e), tau, info)
  open(unit=10, file='results/dlaed4_output.txt', status='replace')
    write(10,*) 'DLAED4 Output:'
    write(10,*) 'Computed tau:', tau
    write(10,*) 'Info:', info
  close(10)

  ! 26. DLAED5
  call DLAED5(1, diag, e, tau, z(1:2))
  open(unit=10, file='results/dlaed5_output.txt', status='replace')
    write(10,*) 'DLAED5 Output:'
    write(10,*) 'Computed tau:', tau
    write(10,*) 'Updated z:', z(1:2)
  close(10)

  ! 27. DLAED6
  call DLAED6(1, .false., diag(1), diag(2), e(1), tau, info)
  open(unit=10, file='results/dlaed6_output.txt', status='replace')
    write(10,*) 'DLAED6 Output:'
    write(10,*) 'Computed tau:', tau
    write(10,*) 'Info:', info
  close(10)

  ! 28. DLAED7
  allocate(work_alloc(4*n_e))
  allocate(iwork_alloc(2*n_e))
  diag = [5.0d0, 4.0d0, 3.0d0, 2.0d0, 1.0d0]
  z(1:n_e) = 1.0d0
  k = n_e
  call DLAED7(n_e, k, 1, diag, e, mat_a, n_e, z(1:n_e), & 
              work_alloc, iwork_alloc, info)
  open(unit=10, file='results/dlaed7_output.txt', status='replace')
    write(10,*) 'DLAED7 Output:'
    write(10,*) 'Updated Eigenvalues:', diag
    write(10,*) 'Updated Eigenvectors:'
    do i = 1, n_e
      write(10, '(5(F8.4,1X))') mat_a(1:n_e,i)
    end do
    write(10,*) 'Info:', info
  close(10)
  deallocate(work_alloc)
  deallocate(iwork_alloc)

  ! 29. DLAED8
  allocate(work_alloc(4*n_e))
  allocate(iwork_alloc(2*n_e))
  diag = [5.0d0, 4.0d0, 3.0d0, 2.0d0, 1.0d0]
  z(1:n_e) = 1.0d0
  k = n_e
  icompq = 0
  call DLAED8(icompq, n_e, diag, e, mat_a, n_e, z(1:n_e), & 
              work_alloc, iwork_alloc, info)
  open(unit=10, file='results/dlaed8_output.txt', status='replace')
    write(10,*) 'DLAED8 Output:'
    write(10,*) 'Updated Eigenvalues:', diag
    write(10,*) 'Updated Eigenvectors:'
    do i = 1, n_e
      write(10, '(5(F8.4,1X))') mat_a(1:n_e,i)
    end do
    write(10,*) 'Info:', info
  close(10)
  deallocate(work_alloc)
  deallocate(iwork_alloc)

  ! 30. DLAED9
  allocate(work_alloc(4*n_e))
  diag = [5.0d0, 4.0d0, 3.0d0, 2.0d0, 1.0d0]
  z(1:n_e) = 1.0d0
  k = n_e
  call DLAED9(n_e, k, 1, diag, e, mat_a, n_e, z(1:n_e), work_alloc, info)
  open(unit=10, file='results/dlaed9_output.txt', status='replace')
    write(10,*) 'DLAED9 Output:'
    write(10,*) 'Updated Eigenvalues:', diag
    write(10,*) 'Updated Eigenvectors:'
    do i = 1, n_e
      write(10, '(5(F8.4,1X))') mat_a(1:n_e,i)
    end do
    write(10,*) 'Info:', info
  close(10)
  deallocate(work_alloc)

  ! 31. DLAEDA
  allocate(work_alloc(4*n_e))
  allocate(iwork_alloc(2*n_e))
  diag = [5.0d0, 4.0d0, 3.0d0, 2.0d0, 1.0d0]
  z(1:n_e) = 1.0d0
  call DLAEDA(n_e, 1, diag, e, mat_a, n_e, z(1:n_e), work_alloc, iwork_alloc, info)
  open(unit=10, file='results/dlaeda_output.txt', status='replace')
    write(10,*) 'DLAEDA Output:'
    write(10,*) 'Info:', info
  close(10)
  deallocate(work_alloc)
  deallocate(iwork_alloc)

  ! 32. DGEMV
  call DGEMV('N', n, n, alpha, mat_a, n, a, 1, beta, b, 1)
  open(unit=10, file='results/dgemv_output.txt', status='replace')
    write(10,*) 'DGEMV Output:'
    write(10,*) 'Resultant B:', b
  close(10)
  a = [1.0d0, 2.0d0, 3.0d0]
  b = [4.0d0, 5.0d0, 6.0d0]

  ! 33. DSTEQR
  allocate(work_alloc(2*n-2))
  diag(1:n) = [5.0d0, 4.0d0, 3.0d0]
  e(1:n-1) = [1.0d0, 1.0d0]
  call DSTEQR('I', n, diag(1:n), e(1:n-1), mat_a, n, work_alloc, info)
  open(unit=10, file='results/dsteqr_output.txt', status='replace')
    write(10,*) 'DSTEQR Output:'
    write(10,*) 'Eigenvalues:', diag(1:n)
    write(10,*) 'Eigenvectors (columns):'
    do i = 1, n
      write(10, '(3(F8.4,1X))') mat_a(1:n,i)
    end do
    write(10,*) 'Info:', info
  close(10)
  deallocate(work_alloc)

  ! 34. DLASCL
  call DLASCL('G', 0, 0, 1.0d0, 2.0d0, n, 1, a, n, info)
  open(unit=10, file='results/dlascl_output.txt', status='replace')
    write(10,*) 'DLASCL Output:'
    write(10,*) 'Scaled A:', a
    write(10,*) 'Info:', info
  close(10)
  a = [1.0d0, 2.0d0, 3.0d0]

  ! 35. DLAEV2
  call DLAEV2(1.0d0, 2.0d0, 3.0d0, f, g, cs, sn)
  open(unit=10, file='results/dlaev2_output.txt', status='replace')
    write(10,*) 'DLAEV2 Output:'
    write(10,*) 'Eigenvalues:', f, g
    write(10,*) 'CS:', cs
    write(10,*) 'SN:', sn
  close(10)

  ! 36. DLASR
  call DLASR('R', 'V', 'F', n, n, a, b, mat_a, n)
  open(unit=10, file='results/dlasr_output.txt', status='replace')
    write(10,*) 'DLASR Output:'
    write(10,*) 'Matrix A after DLASR:'
    do i = 1, n
      write(10, '(3(F8.4,1X))') mat_a(i,:)
    end do
  close(10)

  ! 37. DSWAP
  open(unit=10, file='results/dswap_output.txt', status='replace')
    write(10,*) 'DSWAP Output:'
    write(10,*) 'Original A:', a
    write(10,*) 'Original B:', b
    call DSWAP(n, a, 1, b, 1)
    write(10,*) 'A after DSWAP:', a
    write(10,*) 'B after DSWAP:', b
  close(10)

end program test_all_lapack
