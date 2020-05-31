#include <iostream>
#include <torch/torch.h>
#include <random>
#include <thread>
#include <tuple>

class velocity_kalman
{
	double dt = 0.1;

	torch::Tensor A = torch::tensor
	({
		{ 1.0, dt  },
		{ 0.0, 1.0 },
	});
	torch::Tensor H = torch::tensor({ 1, 0 });
	torch::Tensor Q = torch::tensor
	({
		{ 1, 0 },
		{ 0, 3 },
	});
	double R = 10.0;

	torch::Tensor x = torch::tensor({ 0, 20 });
	torch::Tensor P = 5.0 * torch::eye(2);

public:
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gain(double z)
	{
		torch::Tensor xp = A * x;
		torch::Tensor Pp = A * P * A;
		torch::Tensor K = Pp * H * (1.0/(H * Pp * H + R));

		x = xp + K * (z - H * xp);
		P = Pp - K * H * Pp;
		return std::make_tuple(x, P, K);
	}

private:
	std::default_random_engine rand;
	std::normal_distribution<double> gaussian = std::normal_distribution<double>(0.0, 2.0);
	double pp = 0.0, vp = 80.0;

public:
	double rand_position()
	{
		double w = 0.0 + 10.0 * this->gaussian(this->rand);
		double v = 0.0 + 10.0 * this->gaussian(this->rand);
		double z = pp + vp * dt + v;
		pp = z - v; // true position
		vp = 80.0 + w; // true velocity
		return z;
	}

};


int main()
{
	using namespace std;
	using namespace std::chrono;
	using namespace std::chrono_literals;

	std::default_random_engine rand;
	normal_distribution<double> gaussian(0.0, 2.0);
	velocity_kalman vk;

	high_resolution_clock::now();

	while (true)
	{
		std::this_thread::sleep_for(0.1s);
		double z = vk.rand_position();
		torch::Tensor x, P, K;
		tie(x, P, K) = vk.gain(z);	
		cout << "z:" << z
			<< ", x:" << x
			<< ", P:" << P
			<< ", K:" << K
			<< endl;
	}
}
