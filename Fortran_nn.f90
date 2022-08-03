program neural_network

  use iso_fortran_env, only: int16, int32, real64
  
  implicit none

  integer(int32) :: i
  
  ! Neurons in each layer: input (i), hidden (h), output (o)
  integer(int16), parameter :: n_i = 2, n_h = 5, n_o = 1
  
  ! Number of data samples
  integer(int32), parameter :: n_samples = 4! changed from 4 to 5

  ! learning rate
  real(real64), parameter :: lr = 1.0

  ! Number of iterations
  integer(int32), parameter :: n_iter = 100000

  ! Weights and biases
  real(real64) :: wih(n_h, n_i), who(n_o, n_h), b_h(n_h), b_o(n_o)

  ! Outputs of the hidden and output layers
  real(real64) :: hid(n_h, n_samples), out(n_o, n_samples)
  
  ! Deltas (parts of gradients)
  real(real64) :: delta_o(n_o, n_samples), delta_h(n_h, n_samples)

  ! Inputs and targets
  real(real64) :: x(n_i, n_samples), y(n_o, n_samples)
  real(real64), Dimension(143,5) :: p
  real(real64), Dimension(143,1) :: q
  open(10,file='EOS_x_data.txt')
  read(10,'(5f4.1)') p
  open(11,file='EOS_y_data.txt')
  read(11,'(5f4.1)') q
  x = reshape([p],shape(x))!reshape([0.,0., 0.,1., 1.,0., 1.,1.,1.5], shape(x))!added a "1.5" here
  y = reshape([q], shape(y))!added a "1" here
  
  ! Initialize weights and biases
  call random_number(wih)
  call random_number(b_h)
  call random_number(who)
  call random_number(b_o)
  wih = wih - 0.5
  b_h = b_h - 0.5
  who = who - 0.5
  b_o = b_o - 0.5
  
  do i = 1, n_iter

    ! Forward
    hid = tanh(   lincomb(x,   wih, b_h))
    out = sigmoid(lincomb(hid, who, b_o))
  
    ! Backward
    delta_o = sigmoid_deriv(out) * (out - y)
    delta_h = tanh_deriv(hid) *  matmul(transpose(who), delta_o) 
  
    ! Update weights and biases
    who = who - lr * matmul(delta_o, transpose(hid)) / n_samples
    b_o = b_o - lr * sum(delta_o, 2) / n_samples
    wih = wih - lr * matmul(delta_h, transpose(x)) / n_samples
    b_h = b_h - lr * sum(delta_h, 2) / n_samples
  
    if (mod(i, 10000) == 0) then 
      print *, 'Error: ', sum(abs(out - y)) / (n_o * n_samples)
      print *, 'Prediction: ', sum(abs(out))
    end if 
  end do

contains

  pure function sigmoid(x) result(res)
    real(real64), intent(in) :: x(:,:)
    real(real64) :: res(size(x, 1), size(x, 2))
    res = 1. / (1. + exp(-x)) 
  end function sigmoid

  pure function sigmoid_deriv(x) result(res)
    real(real64), intent(in) :: x(:,:)
    real(real64) :: res(size(x, 1), size(x, 2))
    res = x * (1. - x)
  end function sigmoid_deriv

  pure function tanh_deriv(x) result(res) 
    real(real64), intent(in) :: x(:,:)
    real(real64) :: res(size(x, 1), size(x, 2))
    res = 1. - x ** 2 
  end function tanh_deriv

  pure function lincomb(inp, weights, biases) result (res)
    real(real64), intent(in) :: inp(:,:)
    real(real64), intent(in) :: weights(:,:)
    real(real64), intent(in) :: biases(:)
    real(real64) :: res(size(weights, 1), size(inp, 2))
    res = matmul(weights, inp) + spread(biases, 2, size(inp, 2))
  end function lincomb

end program neural_network
