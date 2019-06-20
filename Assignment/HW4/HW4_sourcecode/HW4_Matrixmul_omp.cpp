#include <iostream>
#include <string>
#include <ctype.h> 
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include <chrono>
#include <ctime>

using namespace std;

double seqcal(double* res, int* tmp_r, int* tmp_c, double* tmp_v, double* Vec, int size) {
	using namespace std::chrono;
	
	steady_clock::time_point t1 = steady_clock::now();

	for (int i = 0; i < size; i++) {
		res[tmp_r[i]] = res[tmp_r[i]] + tmp_v[i] * Vec[tmp_c[i]];
	}

	steady_clock::time_point t2 = steady_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	return time_span.count()*1000;
}

double parallelcal(double* res, int* tmp_r, int* tmp_c, double* tmp_v, double* Vec, int size, int tnums) {
	
	using namespace std::chrono;
	
	steady_clock::time_point t1 = steady_clock::now();
	
	#pragma omp parallel for num_threads(tnums)
	for (int i = 0; i < size; i++) {
		res[tmp_r[i]] = res[tmp_r[i]] + tmp_v[i] * Vec[tmp_c[i]];
	}

	steady_clock::time_point t2 = steady_clock::now();
	duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
	return time_span.count()*1000;
}

double* Createvec(int n, int mod) {
	double *vec = new double[n];
	int i = 0;
	while (i!=n+1) {
		vec[i++] = rand() % mod;
	}
	return vec;
}

int main() {
	FILE* fp;
	char buf[250];
	int rownum, colnum, totnum;
	int tmp_r[10005] = { 0 }, tmp_c[10005] = { 0 };
	double tmp_v[10005] = { 0 };
	double res[100005] = { 0 };
	double *anscopy = new double[sizeof(res)];
	fp = fopen("1138_bus.mtx", "r");
	if (fp != nullptr) {
		fgets(buf, 250, fp);
		cout << buf;
		fscanf(fp, "%d %d %d", &rownum, &colnum, &totnum);
		cout << "Row:" << rownum << endl;
		cout << "Col:" << colnum << endl;
		cout << "Tot:" << totnum << endl;

		double *Vec = Createvec(rownum, 1000);

		for (int i = 0; i < totnum; i++) {
			fscanf(fp, "%d %d  %lf", &tmp_r[i], &tmp_c[i], &tmp_v[i]);
		}
		
		double seq_t = seqcal(res, tmp_r, tmp_c, tmp_v, Vec, totnum);
		cout << "Sequential: " << seq_t << "ms." << endl;
		memcpy(anscopy, res, sizeof(res));
		
		while(1){
			int nums;
			memset(res, 0, sizeof(res));
			cout<<"Parallel with ? threads? (Enter Thread numbers)"<<endl;
			cin >> nums;
			if (nums == -1){
				cout << "Exit Parallel Mode."<<endl;
				break;
			}
			else{
				double par_t = parallelcal(res, tmp_r, tmp_c, tmp_v, Vec, totnum, nums);
				cout << "Parallel: " << par_t << " ms.";
				cout << " (" << seq_t/par_t << " faster ) " << endl;
			}				
		}
	}
	fclose(fp);
	cout << "Result? (Y/N)" << endl;
	char tmp;
	getchar();
	if((tmp = getchar())&&(tmp=='Y'||tmp=='y')){
		for (int i = 0; i < rownum; i++) {
			cout << anscopy[i] << endl;
		}
	}
	return 0;
}
