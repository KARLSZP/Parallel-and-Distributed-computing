#include <stdio.h>
#include <omp.h>
#define queue_size 10

int in = 0, out = 0;	// ���ס���β���Ӷ���ȡ���Ӷ�β���� 
void producer(int); // �����ߺ��� 
void consumer(); // �����ߺ��� 
omp_lock_t pro_lock;
omp_lock_t con_lock;

/* �ж϶ӿ� */
int queue_is_empty() {
	return (in == out);
}
int queue_is_full() {
	return (out == ((in + 1) % queue_size));
}

/* ���߳� */
int main() {
	int nums_pro, nums_con;
	scanf("%d%d", &nums_pro, &nums_con);
	#pragma omp parallel sections
	{
		#pragma omp section
		producer(nums_pro);

		#pragma omp section
		consumer();
	}
	return 0;
}

void producer(int nums_pro) {
	#pragma omp parallel for num_threads(nums_pro)
	for(int i = 0;i < nums_pro;i++){
		while (1) {
			omp_set_lock(&pro_lock);
			while (queue_is_full()) {}
			printf("producer [%d]:\t no.%d item produced. \n", omp_get_thread_num(), in);
			in = (in + 1) % queue_size;
			#pragma omp flush
			omp_unset_lock(&pro_lock);
		}		
	}
}

void consumer() {
	
	while (1) {
		omp_set_lock(&con_lock);
		while (queue_is_empty()) {}
		printf("consumer [%d]:\t no.%d item consumed. \n", omp_get_thread_num(), out);
		out = (out + 1) % queue_size;
		#pragma omp flush
		omp_unset_lock(&con_lock);
	}
}
