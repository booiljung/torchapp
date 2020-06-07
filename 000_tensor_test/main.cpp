#include <iostream>
#include <random>
#include <thread>
#include <tuple>
#include <torch/torch.h>

int main(int argc, char *argv[])
{
	using namespace std;
	using namespace std::chrono;
	using namespace std::chrono_literals;
	using namespace torch;
	using namespace at;

	auto a = torch::tensor
	({
		{ 1.0, 0.0 },
		{ 0.0, 1.0 },
	}, at::dtype(at::kDouble));

	auto b = torch::tensor
	({
		{ 1.0 },
		{ 2.0 }
	}, at::dtype(at::kDouble));

	auto c = torch::matmul(a, b);

	cout << a << endl;
	cout << b << endl;
	cout << c << endl;
}
