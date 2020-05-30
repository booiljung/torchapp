#include <iostream>
#include <torch/torch.h>
#include <random>
#include <thread>

class simple_kalman
{
	double A = 1.0,
		H = 1.0,
		Q = 0.0,
		R = 3.0;

	double x = 0,
		P = 20;


public:
	double gain(double z)
	{
		double xp = A * x;
		double Pp = A * P * A;
		double K = Pp * H * (1.0/(H * Pp * H + R));

		x = xp + K * (z - H * xp);
		P = Pp - K * H * Pp;
		return x;
	}
};


int main() {
	using namespace std;
	using namespace std::chrono;
	using namespace std::chrono_literals;

	std::default_random_engine rand;
	normal_distribution<double> gaussian(0.0, 2.0);
	simple_kalman sk;

	auto start = high_resolution_clock::now();
	std::this_thread::sleep_for(1s);

	double z = 5.0;
	double d = -0.001;
	while (true)
	{
		std::this_thread::sleep_for(0.1s);		
		double v = gaussian(rand);
		z += d;
		if (d < 0.0 && z <= 0.0)
			d = +0.001;
		else if (0.0 < d && 5.0 <= d)
			d = -0.001;
		double x = sk.gain(z + v);
		cout << z << ", " << z + v << ", " << x << endl;
	}
}
