#include <iostream>
#include <torch/torch.h>
#include <random>
#include <thread>
#include <tuple>

#include "qcustomplot.h"

class position_kalman
{

	double dt = 0.1;

	torch::Tensor A = torch::tensor
	({
		{ 1.0, dt  },
		{ 0.0, 1.0 },
	}, at::dtype(at::kDouble));
	torch::Tensor H = torch::tensor
	({
		{ 0.0, 1.0 }
	}, at::dtype(at::kDouble));
	torch::Tensor Q = torch::tensor
	({
		{ 1.0, 0.0 },
		{ 0.0, 3.0 },
	}, at::dtype(at::kDouble));
	double R = 10.0;

	torch::Tensor x = torch::tensor({
			{ 0.0 },
			{ 20.0 }
	}, at::dtype(at::kDouble));

	torch::Tensor P = 5.0 * torch::eye(2, at::dtype(at::kDouble));

public:
	torch::Tensor gain(double z)
	{
		torch::Tensor xp = matmul(A, x);
		torch::Tensor Pp = matmul(matmul(A, P), torch::transpose(A, 0, 1)) + Q;
		torch::Tensor K = matmul(
			matmul(
				Pp, torch::transpose(H, 0, 1)
			), torch::inverse(
				matmul(
					matmul(H, Pp), torch::transpose(H, 0, 1)
				) + R
			)
		);
		x = xp + matmul(K, (z - matmul(H, xp)));
		P = Pp - matmul(matmul(K, H), Pp);
		return x;
	}

private:
	std::default_random_engine rand;
	std::normal_distribution<double> gaussian = std::normal_distribution<double>(0.0, 2.0);
	double vp = 0.0, pp = 80.0;

public:
	double rand_velocity()
	{
		double v = 0.0 + 10.0 * this->gaussian(this->rand);
		pp = pp + vp * dt; // true position
		vp = 80.0 + v; // true velocity
		return vp;
	}

};


int main(int argc, char *argv[])
{
	using namespace std;
	using namespace std::chrono;
	using namespace std::chrono_literals;
	using namespace at;

	std::default_random_engine rand;
	normal_distribution<double> gaussian(0.0, 10.0);
	position_kalman pk;

	high_resolution_clock::now();

	QVector<double> history_t;
	QVector<double> history_z;
	QVector<double> history_x00;
	QVector<double> history_x01;

	double t = 0.0;
	double dt = 0.1;
	for (int i = 0; i < 100; i++)
	{
		history_t.push_back(t);

		double z = pk.rand_velocity();
		history_z.push_back(z);

		torch::Tensor x = pk.gain(z);	
		auto x_accessor = x.accessor<double, 2>();
		history_x00.push_back(x_accessor[0][0]-20);
		history_x01.push_back(x_accessor[0][1]-10);

		cout << "=============" << endl
			<< "z:" << z << endl
			<< "x:" << x << endl;

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
	customPlot.resize(1600, 1200);
	customPlot.xAxis->setLabel("time");
	customPlot.yAxis->setLabel("value");
	customPlot.xAxis->setRange(0, 100*dt);
	customPlot.yAxis->setRange(-200, 1200);

	// create graph and assign data to it:
	QPen pen;

	customPlot.addGraph();
	pen.setColor(Qt::GlobalColor::black);
	customPlot.graph()->setPen(pen);
	customPlot.graph()->setData(history_t, history_z);
	customPlot.graph()->setName("z");

	customPlot.addGraph();
	pen.setColor(Qt::GlobalColor::blue);
	customPlot.graph()->setPen(pen);
	customPlot.graph()->setData(history_t, history_x00);
	customPlot.graph()->setName("x00");

	customPlot.addGraph();
	pen.setColor(Qt::GlobalColor::cyan);
	customPlot.graph()->setPen(pen);
	customPlot.graph()->setData(history_t, history_x01);
	customPlot.graph()->setName("x11");

	customPlot.legend->setVisible(true);
	customPlot.legend->setRowSpacing(-3);

	customPlot.axisRect()->setupFullAxesBox();
	customPlot.replot();
	customPlot.show();

	app.exec();	
}
