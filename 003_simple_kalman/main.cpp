#include <iostream>
#include <random>
#include <thread>
#include <tuple>

#include <torch/torch.h>
#include <QtCore/QVector>
#include "qcustomplot.h"

class simple_kalman
{
	double A = 1.0,
		H = 1.0,
		Q = 0.0, // 크만 민감, 작으면 둔감.
		R = 3.0; // 크면 둔감, 작으면 민감.

	double x = 0,
		P = 50;


public:
	std::tuple<double, double, double> gain(double z)
	{
		double xp = A * x;
		double Pp = A * P * A;
		double K = Pp * H * (1.0/(H * Pp * H + R));

		x = xp + K * (z - H * xp);
		P = Pp - K * H * Pp;
		return std::make_tuple(x, P, K);
	}
};


int main(int argc, char *argv[])
{
	using namespace std;
	using namespace std::chrono;
	using namespace std::chrono_literals;

	// for normal distribution noise
	normal_distribution<double> gaussian(0.0, 0.5);
	std::default_random_engine rand;

	simple_kalman sk;

	high_resolution_clock::now();

	QVector<double> history_t;
	QVector<double> history_z;
	QVector<double> history_zv;
	QVector<double> history_x;
	QVector<double> history_P;
	QVector<double> history_K;

	double t = 0.0;
	double dt = 0.1;
	double z = 5.0;
	double d = -0.01;
	for (int i = 0; i < 1000; i++)
	{
		double v = gaussian(rand);
		double x, P, K;
		tie(x, P, K) = sk.gain(z + v);
		history_t.push_back(t);
		history_z.push_back(z);
		history_zv.push_back(z+v);
		history_x.push_back(x);
		history_P.push_back(P);
		history_K.push_back(K);
		cout << "z:" << z
			<< ", z+v:" << z + v
			<< ", x:" << x
			<< ", P:" << P
			<< ", K:" << K
			<< endl;
		t += dt;
		z += d;
		if (d < 0.0 && z <= 0.0)
			d = +0.01;
		else if (0.0 < d && 5.0 <= d)
			d = -0.01;
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
	customPlot.xAxis->setRange(0, 1000*dt);
	customPlot.yAxis->setRange(-2, 7);

	// create graph and assign data to it:
	QPen pen;

	customPlot.addGraph();
	pen.setColor(Qt::GlobalColor::black);
	customPlot.graph()->setPen(pen);
	customPlot.graph()->setData(history_t, history_z);

	customPlot.addGraph();
	pen.setColor(Qt::GlobalColor::blue);
	customPlot.graph()->setPen(pen);
	customPlot.graph()->setData(history_t, history_zv);

	customPlot.addGraph();
	pen.setColor(Qt::GlobalColor::red);
	customPlot.graph()->setPen(pen);
	customPlot.graph()->setData(history_t, history_x);

	//customPlot.addGraph();
	//customPlot.graph()->setData(history_t, history_P);
	//customPlot.addGraph();
	//customPlot.graph()->setData(history_t, history_K);

	//, history_zv, history_x, history_P, history_K

	
	customPlot.replot();
	customPlot.show();

	app.exec();
}
