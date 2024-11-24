! src/test_lapack.f90
program test_all_lapack
  implicit none
  integer :: i, info
  real(8) :: dlamch_val, dnrm2_val, dlapy2_val, dlassq_val
  real(8), dimension(3) :: a, b, c, work
  real(8), dimension(3,3) :: mat_a, mat_b, mat_c
  real(8), dimension(5) :: e, diag, offdiag, work5
  real(8) :: cs, sn, f, g, tau
  integer :: iamax_val
  real(8), dimension(4) :: z
  integer :: ldz = 4
  integer :: n = 3
  integer :: n_e = 5
  integer :: n_z = 4
  integer :: one = 1
  real(8), parameter :: alpha = 2.0d0, beta = 0.5d0
  real(8), dimension(3,2) :: mat_3x2
  real(8), dimension(2,3) :: mat_2x3
  integer :: n_rows, n_cols
  character(len=20) :: filename
  integer :: k, nl, nr, icompq, lwork
  real(8), allocatable :: work_alloc(:)
  integer, dimension(5) :: iwork
  logical :: bwed
  
  ! Initialize arrays and matrices
  a = [1.0d0, 2.0d0, 3.0d0]
  b = [4.0d0, 5.0d0, 6.0d0]
  c = [0.0d0, 0.0d0, 0.0d0]
  work = [0.0d0, 0.0d0, 0.0d0]

  mat_a = reshape([1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0, 6.0d0, 7.0d0, 8.0d0, 9.0d0], shape(mat_a))
  mat_b = reshape([9.0d0, 8.0d0, 7.0d0, 6.0d0, 5.0d0, 4.0d0, 3.0d0, 2.0d0, 1.0d0], shape(mat_b))
  mat_c = 0.0d0
  e = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 0.0d0]
  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0]
  offdiag = [1.0d0, 1.0d0, 1.0d0, 1.0d0, 0.0d0]
  work5 = 0.0d0
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0]

  mat_3x2 = reshape([1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0, 6.0d0], shape(mat_3x2))
  mat_2x3 = reshape([1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0, 6.0d0], shape(mat_2x3))

  iwork = 0

  ! Test functions and write results
  
  ! 1. DLAMCH
  dlamch_val = DLAMCH('E')
  open(unit=10, file='results/dlamch_output.txt', status='replace')
  write(10,*) 'DLAMCH(''E'') = ', dlamch_val
  close(10)

  ! 2. DSBTRD
  call DSBTRD('U', n, 1, mat_a, n, diag, offdiag, work, info)
  open(unit=10, file='results/dsbtrd_output.txt', status='replace')
    write(10,*) 'DSBTRD Output:'
    write(10,*) 'Diagonal:', diag
    write(10,*) 'Off-Diagonal:', offdiag
    write(10,*) 'Info:', info
  close(10)

  ! 3. DLASET
  call DLASET('U', n, n, alpha, beta, mat_c, n)
  open(unit=10, file='results/dlaset_output.txt', status='replace')
    write(10,*) 'DLASET Output:'
    write(10,*) 'Matrix C:', mat_c
  close(10)

  mat_c = 0.0d0 ! Reset mat_c for next test

  ! 4. DLARTG
  call DLARTG(1.0d0, 2.0d0, cs, sn, f)
  open(unit=10, file='results/dlartg_output.txt', status='replace')
    write(10,*) 'DLARTG Output:'
    write(10,*) 'CS:', cs
    write(10,*) 'SN:', sn
    write(10,*) 'F:', f
  close(10)

  ! 5. DROT
  call DROT(n, a, 1, b, 1, cs, sn)
  open(unit=10, file='results/drot_output.txt', status='replace')
  write(10,*) 'DROT Output:'
  write(10,*) 'A after rotation:', a
  write(10,*) 'B after rotation:', b
  close(10)

  ! Reinitialize a, b for subsequent tests
  a = [1.0d0, 2.0d0, 3.0d0]
  b = [4.0d0, 5.0d0, 6.0d0]

  ! 6. DLARTV
  call DLARTV(n, z(1:3), 1, z(2:4), 1, cs, sn, 3)
  open(unit=10, file='results/dlartv_output.txt', status='replace')
  write(10,*) 'DLARTV Output:'
  write(10,*) 'Z after transformation:', z
  close(10)

  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] ! Reinitialize z

  ! 7. DLARGV
  call DLARGV(n, a, 1, z(1:3), 1, work, 3)
  open(unit=10, file='results/dlargv_output.txt', status='replace')
  write(10,*) 'DLARGV Output:'
  write(10,*) 'A after transformation:', a
  write(10,*) 'Z after transformation:', z(1:3)
  write(10,*) 'Work array:', work
  close(10)

  a = [1.0d0, 2.0d0, 3.0d0] ! Reinitialize a
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] ! Reinitialize z
  work = [0.0d0, 0.0d0, 0.0d0]

  ! 8. DLAR2V
  call DLAR2V(n, a(1:2), b(1:2), c(1:2), n, cs, sn )
  open(unit=10, file='results/dlar2v_output.txt', status='replace')
    write(10,*) 'DLAR2V Output:'
    write(10,*) 'A after transformation:', a(1:2)
    write(10,*) 'B after transformation:', b(1:2)
    write(10,*) 'C after transformation:', c(1:2)
    write(10,*) 'CS:', cs
    write(10,*) 'SN:', sn
  close(10)

  a = [1.0d0, 2.0d0, 3.0d0] ! Reinitialize a
  b = [4.0d0, 5.0d0, 6.0d0]
  c = [0.0d0, 0.0d0, 0.0d0]

  ! 9. DSTEDC
  icompq = 1 ! Calculate eigenvectors
  lwork = 6*n_e
  allocate(work_alloc(lwork))
  work_alloc = 0
  call DSTEDC('I', n_e, diag, e, mat_a, n_e, work_alloc, lwork, iwork, 5*n_e, info)
    open(unit=10, file='results/dstedc_output.txt', status='replace')
      write(10,*) 'DSTEDC Output:'
      write(10,*) 'Eigenvalues:', diag
      write(10,*) 'Eigenvectors:', mat_a
      write(10,*) 'Info:', info
   close(10)
   deallocate(work_alloc)
   mat_a = reshape([1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0, 6.0d0, 7.0d0, 8.0d0, 9.0d0], shape(mat_a)) !reinitialize

  ! 10. DLANST
  dlassq_val = DLANST('M', n, diag, offdiag)
  open(unit=10, file='results/dlanst_output.txt', status='replace')
    write(10,*) 'DLANST Output:'
    write(10,*) 'Max abs sum:', dlassq_val
  close(10)

  ! 11. DLASSQ
  dlassq_val = DLASSQ(n, a, one, 1.0d0, 0.0d0)
  open(unit=10, file='results/dlassq_output.txt', status='replace')
    write(10,*) 'DLASSQ Output:'
    write(10,*) 'Value:', dlassq_val
  close(10)

  a = [1.0d0, 2.0d0, 3.0d0] ! Reinitialize a

  ! 12. DLAED0
  icompq = 0 ! No eigenvectors
  lwork = 3*n_e + n_e*n_e
  allocate(work_alloc(lwork))
  work_alloc = 0
  iwork = 0
  call DLAED0(icompq, n_e, 1, n_e, mat_a, n_e, diag, e, z, n_z, work_alloc, lwork, iwork, info)
    open(unit=10, file='results/dlaed0_output.txt', status='replace')
      write(10,*) 'DLAED0 Output:'
      write(10,*) 'Info:', info
    close(10)
  deallocate(work_alloc)

  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize
  e = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 0.0d0]!reinitialize
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] !reinitialize

  ! 13. DCOPY
  call DCOPY(n, a, 1, c, 1)
  open(unit=10, file='results/dcopy_output.txt', status='replace')
    write(10,*) 'DCOPY Output:'
    write(10,*) 'Copied array C:', c
  close(10)

  c = [0.0d0, 0.0d0, 0.0d0] ! Reset c

  ! 14. DGEMM
  n_rows = size(mat_3x2, 1)
  n_cols = size(mat_2x3, 2)
  call DGEMM('N', 'N', n_rows, n_cols, 2, alpha, mat_3x2, 3, mat_2x3, 2, beta, mat_c, 3)
  open(unit=10, file='results/dgemm_output.txt', status='replace')
    write(10,*) 'DGEMM Output:'
    write(10,*) 'Resultant Matrix C:', mat_c
  close(10)

  mat_c = 0.0d0 !reset mat_c

  ! 15. DLACPY
  call DLACPY('A', 3, 3, mat_a, 3, mat_c, 3)
  open(unit=10, file='results/dlacpy_output.txt', status='replace')
    write(10,*) 'DLACPY Output:'
    write(10,*) 'Copied Matrix C:', mat_c
  close(10)

  mat_c = 0.0d0 ! Reset mat_c

  ! 16. DLAED1
  k = 2 
  nl = 3
  nr = 5-k-1
  lwork = 4*5
  allocate(work_alloc(lwork))
  work_alloc = 0
  call DLAED1(n_e-1, diag, e, nl, nr, z, ldz, iwork, work_alloc, info)
    open(unit=10, file='results/dlaed1_output.txt', status='replace')
      write(10,*) 'DLAED1 Output:'
      write(10,*) 'Z:', z
      write(10,*) 'Diag:', diag
      write(10,*) 'E:', e
      write(10,*) 'Info:', info
    close(10)
  deallocate(work_alloc)
  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize
  e = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 0.0d0]!reinitialize
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] !reinitialize

  ! 17. DLAED2
  k = 2
  nl = 3
  lwork = 4*5
  allocate(work_alloc(lwork))
  work_alloc = 0
  call DLAED2(nl, nr-1, k, n_e-1, diag, z, ldz, diag, e, work_alloc, info)
    open(unit=10, file='results/dlaed2_output.txt', status='replace')
      write(10,*) 'DLAED2 Output:'
      write(10,*) 'Z:', z
      write(10,*) 'Diag:', diag
      write(10,*) 'E:', e
      write(10,*) 'Info:', info
    close(10)
  deallocate(work_alloc)
  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize
  e = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 0.0d0]!reinitialize
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] !reinitialize

  ! 18. IDAMAX
  iamax_val = IDAMAX(n, a, 1)
  open(unit=10, file='results/idamax_output.txt', status='replace')
    write(10,*) 'IDAMAX Output:'
    write(10,*) 'Index of max element in A:', iamax_val
  close(10)

  ! 19. DLAPY2
  dlapy2_val = DLAPY2(1.0d0, 2.0d0)
  open(unit=10, file='results/dlapy2_output.txt', status='replace')
    write(10,*) 'DLAPY2 Output:'
    write(10,*) 'Result:', dlapy2_val
  close(10)

  ! 20. DLAMRG
  nl = 2
  nr = 3
  iwork(1) = 1
  iwork(2) = 3
  iwork(3) = 2
  iwork(4) = 4
  iwork(5) = 5
  allocate(work_alloc(2*5))
  work_alloc = 0
  call DLAMRG(nl, nr, diag, 1, -1, iwork)
    open(unit=10, file='results/dlamrg_output.txt', status='replace')
      write(10,*) 'DLAMRG Output:'
      write(10,*) 'Diag reordered:', diag
      write(10,*) 'Index array (iwork):', iwork
    close(10)
  deallocate(work_alloc)

  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize

  ! 21. DSCAL
  call DSCAL(n, 2.0d0, a, 1)
  open(unit=10, file='results/dscal_output.txt', status='replace')
    write(10,*) 'DSCAL Output:'
    write(10,*) 'Scaled array A:', a
  close(10)

  a = [1.0d0, 2.0d0, 3.0d0] ! Reinitialize a

  ! 22. DLAED3
  k=2
  nl=3
  nr=1
  lwork = 4*5
  allocate(work_alloc(lwork))
  work_alloc = 0
  icompq = 0
  call DLAED3(nl, k, nr, n_e-1, diag, e, z, ldz, work_alloc, lwork, iwork, info)
  open(unit=10,file='results/dlaed3_output.txt', status='replace')
     write(10,*) 'DLAED3 Output:'
     write(10,*) 'Diag: ', diag
     write(10,*) 'E: ', e
     write(10,*) 'Z: ', z
     write(10,*) 'Info: ', info
  close(10)
  deallocate(work_alloc)
  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize
  e = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 0.0d0]!reinitialize
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] !reinitialize

  ! 23. DLAMC3
  dlamch_val = DLAMC3(1.0d0, 2.0d0)
    open(unit=10, file='results/dlamc3_output.txt', status='replace')
      write(10,*) 'DLAMC3 Output:'
      write(10,*) 'Value:', dlamch_val
    close(10)

  ! 24. DNRM2
  dnrm2_val = DNRM2(n, a, 1)
    open(unit=10, file='results/dnrm2_output.txt', status='replace')
      write(10,*) 'DNRM2 Output:'
      write(10,*) '2-Norm of A:', dnrm2_val
    close(10)

  ! 25. DLAED4
  k = 2
  lwork = 4*5
  allocate(work_alloc(lwork))
  work_alloc = 0
  iwork(1) = 1
  iwork(2) = 1
  iwork(3) = 1
  iwork(4) = 1
  iwork(5) = 1
  call DLAED4(n_e-1, k, diag, e, z, ldz, work_alloc, iwork, info)
  open(unit=10,file='results/dlaed4_output.txt', status='replace')
     write(10,*) 'DLAED4 Output:'
     write(10,*) 'Diag: ', diag
     write(10,*) 'E: ', e
     write(10,*) 'Z: ', z
     write(10,*) 'Info: ', info
  close(10)
  deallocate(work_alloc)
  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize
  e = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 0.0d0]!reinitialize
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] !reinitialize

  ! 26. DLAED5
  k = 2
  lwork = 4*5
  allocate(work_alloc(lwork))
  work_alloc = 0
  call DLAED5(k, diag, e, z, ldz, work_alloc(1:n_e))
  open(unit=10,file='results/dlaed5_output.txt', status='replace')
     write(10,*) 'DLAED5 Output:'
     write(10,*) 'Diag: ', diag
     write(10,*) 'E: ', e
     write(10,*) 'Z: ', z
  close(10)
  deallocate(work_alloc)
  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize
  e = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 0.0d0]!reinitialize
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] !reinitialize

  ! 27. DLAED6
  k = 2
  lwork = 4*5
  allocate(work_alloc(lwork))
  work_alloc = 0
  call DLAED6(n_e-1, k, iwork, work_alloc(1:n_e), diag, e, z, ldz, info)
  open(unit=10,file='results/dlaed6_output.txt', status='replace')
     write(10,*) 'DLAED6 Output:'
     write(10,*) 'Diag: ', diag
     write(10,*) 'E: ', e
     write(10,*) 'Z: ', z
     write(10,*) 'Info: ', info
  close(10)
  deallocate(work_alloc)
  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize
  e = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 0.0d0]!reinitialize
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] !reinitialize

  ! 28. DLAED7
  icompq = 0
  k = 2
  nl = 3
  nr = 1
  lwork = 4*5
  allocate(work_alloc(lwork))
  work_alloc = 0
  call DLAED7(icompq, n_e-1, nl, nr, diag, z, ldz, iwork, work_alloc, info)
  open(unit=10,file='results/dlaed7_output.txt', status='replace')
     write(10,*) 'DLAED7 Output:'
     write(10,*) 'Diag: ', diag
     write(10,*) 'Z: ', z
     write(10,*) 'Info: ', info
  close(10)
  deallocate(work_alloc)
  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] !reinitialize

  ! 29. DLAED8
  k = 2
  nl = 3
  nr = 1
  lwork = 4*5
  allocate(work_alloc(lwork))
  work_alloc = 0
  call DLAED8(icompq, n_e-1, nl, nr, diag, z, ldz, work_alloc(1:n_e), info)
    open(unit=10,file='results/dlaed8_output.txt', status='replace')
      write(10,*) 'DLAED8 Output:'
      write(10,*) 'Z: ', z
      write(10,*) 'Info: ', info
    close(10)
  deallocate(work_alloc)
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] !reinitialize
  
  ! 30. DLAED9
  k = 2
  nl = 3
  nr = 1
  lwork = 4*5
  allocate(work_alloc(lwork))
  work_alloc = 0
  call DLAED9(n_e-1,k,nl,nr, diag, z, ldz, work_alloc, lwork, iwork, info)
    open(unit=10,file='results/dlaed9_output.txt', status='replace')
      write(10,*) 'DLAED9 Output:'
      write(10,*) 'Diag: ', diag
      write(10,*) 'Z: ', z
      write(10,*) 'Info: ', info
    close(10)
  deallocate(work_alloc)
  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] !reinitialize

  ! 31. DLAEDA
  nl = 3
  nr = 1
  lwork = 4*5
  allocate(work_alloc(lwork))
  work_alloc = 0
  iwork(1) = 1
  iwork(2) = 1
  iwork(3) = 1
  iwork(4) = 1
  iwork(5) = 1
  call DLAEDA(n_e-1, nl, nr, icompq, iwork, diag, z, ldz, work_alloc, info)
  open(unit=10,file='results/dlaeda_output.txt', status='replace')
     write(10,*) 'DLAEDA Output:'
     write(10,*) 'Diag: ', diag
     write(10,*) 'Z: ', z
     write(10,*) 'Info: ', info
  close(10)
  deallocate(work_alloc)
  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize
  z = [1.0d0, 2.0d0, 3.0d0, 4.0d0] !reinitialize

  ! 32. DGEMV
  call DGEMV('N', n, n, alpha, mat_a, n, a, 1, beta, b, 1)
  open(unit=10, file='results/dgemv_output.txt', status='replace')
    write(10,*) 'DGEMV Output:'
    write(10,*) 'Resultant vector B:', b
  close(10)

  a = [1.0d0, 2.0d0, 3.0d0] ! Reinitialize a
  b = [4.0d0, 5.0d0, 6.0d0] ! Reinitialize b

  ! 33. DSTEQR
  lwork = 6*n
  allocate(work_alloc(lwork))
  work_alloc = 0.0d0
  call DSTEQR('I', n, diag(1:n), e(1:n-1), mat_a, n, work_alloc, info)
  open(unit=10, file='results/dsteqr_output.txt', status='replace')
    write(10,*) 'DSTEQR Output:'
    write(10,*) 'Eigenvalues:', diag(1:n)
    write(10,*) 'Eigenvectors:', mat_a
    write(10,*) 'Info:', info
  close(10)
  deallocate(work_alloc)
  diag = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0] !reinitialize
  e = [1.0d0, 2.0d0, 3.0d0, 4.0d0, 0.0d0]!reinitialize
  mat_a = reshape([1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0, 6.0d0, 7.0d0, 8.0d0, 9.0d0], shape(mat_a)) !reinitialize

  ! 34. DLASCL
  call DLASCL('G', 0, 0, 1.0d0, 2.0d0, n, 1, a, n, info)
  open(unit=10, file='results/dlascl_output.txt', status='replace')
    write(10,*) 'DLASCL Output:'
    write(10,*) 'Scaled array A:', a
    write(10,*) 'Info:', info
  close(10)

  a = [1.0d0, 2.0d0, 3.0d0] ! Reinitialize a

  ! 35. DLAEV2
  call DLAEV2(1.0d0, 2.0d0, 3.0d0, cs, sn, f)
  open(unit=10, file='results/dlaev2_output.txt', status='replace')
    write(10,*) 'DLAEV2 Output:'
    write(10,*) 'CS:', cs
    write(10,*) 'SN:', sn
    write(10,*) 'F:', f
  close(10)
