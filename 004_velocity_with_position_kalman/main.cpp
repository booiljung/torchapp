#include <iostream>
#include <random>
#include <thread>
#include <tuple>
#include <torch/torch.h>

#include "qcustomplot.h"

class velocity_kalman
{
	double dt = 0.1;

	torch::Tensor A = torch::tensor
	({		
		{ 1.0, dt  },
		{ 0.0, 1.0 },
	}, at::kDouble);
	torch::Tensor H = torch::tensor({{ 1, 0 }}, at::kDouble);
	torch::Tensor Q = torch::tensor
	({
		{ 1, 0 },
		{ 0, 3 },
	}, at::kDouble);
	double R = 10.0;

	torch::Tensor x = torch::tensor({{ 0, 20 }}, at::kDouble);
	torch::Tensor P = 5.0 * torch::eye(2, at::kDouble);

public:
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> gain(double z)
	{
		torch::Tensor xp = A * x;
		torch::Tensor Pp = A * P * torch::transpose(A, 0, 1) + Q;
		torch::Tensor K = Pp * torch::transpose(H, 0, 1) * torch::inverse((H * Pp * torch::transpose(H, 0, 1) + R));

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


int main(int argc, char *argv[])
{
	using namespace std;
	using namespace std::chrono;
	using namespace std::chrono_literals;
	using namespace at;

	std::default_random_engine rand;
	normal_distribution<double> gaussian(0.0, 2.0);
	velocity_kalman vk;

	high_resolution_clock::now();

	QVector<double> history_t;
	QVector<double> history_z;
	QVector<double> history_zv;
	QVector<double> history_x0;
	QVector<double> history_x1;

	double t = 0.0;
	double dt = 0.1;
	for (int i = 0; i < 100; i++)
	{
		double z = vk.rand_position();
		torch::Tensor x, P, K;
		tie(x, P, K) = vk.gain(z);
		history_t.push_back(t);
		history_z.push_back(z);
		auto x_accessor = x.accessor<double, 2>();
		history_x0.push_back(x_accessor[0][0]);
		history_x1.push_back(x_accessor[0][1]);
		cout << "=============" << endl
			<< "z:" << z << endl
			<< "x:" << x << endl
			<< "P:" << P << endl
			<< "K:" << K << endl;
		t += dt;
	}

	#ifdef Q_OS_ANDROID
    QApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
#endif
	
	//Q_INIT_RESOURCE(application);

	QApplication app(argc, argv);
    QCoreApplication::setOrganizationName("QtProject");
    QCoreApplication::setApplicationName("Application Example");
    QCoreApplication::setApplicationVersion(QT_VERSION_STR);

	QCustomPlot customPlot;
	customPlot.resize(1600, 600);
	customPlot.xAxis->setLabel("time");
	customPlot.yAxis->setLabel("value");
	customPlot.xAxis->setRange(0, 100*dt);
	customPlot.yAxis->setRange(-50, 1000);

	// create graph and assign data to it:
	QPen pen;

	customPlot.addGraph();
	pen.setColor(Qt::GlobalColor::black);
	customPlot.graph()->setPen(pen);
	customPlot.graph()->setData(history_t, history_z);

	customPlot.addGraph();
	pen.setColor(Qt::GlobalColor::red);
	customPlot.graph()->setPen(pen);
	customPlot.graph()->setData(history_t, history_zv);

	customPlot.addGraph();
	pen.setColor(Qt::GlobalColor::blue);
	customPlot.graph()->setPen(pen);
	customPlot.graph()->setData(history_t, history_x0);

	customPlot.addGraph();
	pen.setColor(Qt::GlobalColor::cyan);
	customPlot.graph()->setPen(pen);
	customPlot.graph()->setData(history_t, history_x1);

	//customPlot.addGraph();
	//customPlot.graph()->setData(history_t, history_P);
	//customPlot.addGraph();
	//customPlot.graph()->setData(history_t, history_K);

	//, history_zv, history_x, history_P, history_K

	
	customPlot.replot();
	customPlot.show();

	app.exec();
}
