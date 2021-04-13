#include <iostream>
#include <fstream>
#include <cmath>
#include <gsl/gsl_rng.h>
#include <time.h>
using namespace std;

const int max_side=60;

//Prototype declaration
int remainder(int a, int b);
void direct_translator(int &i, int &j, int k, int side);
void inverse_translator(int i, int j, int &k, int side);
void build_neighbours_array(int side, int neighbours[][max_side*max_side]);
void build_T_array(int howManyTs, double T_ini, double T_step, double T_array[]);
void initialize_uniform_system(gsl_rng* r, int system[], int side);
void initialize_random_system(gsl_rng* r, int system[], int side);
int index_to_delta_E(int index);
int delta_E_to_index(int delta_E);
void build_minimums_array(double T, double minimums_array[]);
void evolve_one_pMC(gsl_rng* r, int side, double minimums_array[], int narr[][max_side*max_side], int system[]);
void evolve_N_pMC(int N, gsl_rng* r, int side, double minimums_array[], int narr[][max_side*max_side], int system[]);
double measure_magnetization(int system[], int side);
void normalize_measurements_array(double measurements[], int length, int n_measurements);
void print_grid(int system[], int side);


//Sobre las variables que voy a utilizar. El sistema lo codifico en un array unidimensional de longitud N, de modo que para un grid cuadrado de spines unidimensional, se tiene que el lado de dicha parrilla, side, es side*side=N. A menudo, me será útil pensar en el sistema como una parrilla bidimensional, por lo que es conveniente establecer una relación entre el etiquetado que usamos en el sistema extendido en un array bidimensional de longitud N, y la parrilla bidimensional. La numeración va a ser tal que los side primeros spines del array unidimensional correspondan con la primera fila de la parrilla bidimensional, los siguiente side spines del array unidimensional correspondan con la segunda fila de la parrilla bidimensional, y en general, (teniendo en cuenta que en C++ los arrays empiezan a indexar en 0) que la fila n-ésima abarca los spines que van entre n*side y -1+((n+1)*side), con n=0,...,side-1. Así pues, el spín k-ésimo en el array unidimensional es el spin (i,j)-ésimo en la parrilla bidimensional, con i=floor(k/side) y j=k%side, con el indexado de C++. Así pues, los dos vecinos a derechas e izquierda del spin k-ésimo del array unidimensonal son el (k-1)-ésimo y el (k+1)-ésimo, si el spin NO está en un extremo de fila, esto es, si k%side!=0.

//Sobre la inicialización uniforme: Si inicializo el sistema de forma aleatoria, no habŕa ningún problema a temperaturas altas, pero a temperaturas bajas se puede necesitar un tiempo de termalización prácticamente infinito, si entendemos que la termalización se consigue (a temperaturas muy bajas) cuando todo el sistema está alineado paralelamente, esto es, todos los spines apuntando hacia arriba (o hacia abajo) y la magnetización se hace igual a 1. Para explicar esto consideremos la siguiente situación: el sistema ha llegado a un estado en que la mitad izquierda de los spines estan en -1 y la mitad derecha de los spines están en el estado +1, a temperatura T=0.05. Obviamente, cualquiera de los spines que NO estén situados en la frontera entre ambas fases magnetizadas antiparalelamente no cambiarán el estado de su spin, pues dicha probabilidad es prácticamente nula de acuerdo con el algoritmo de metropolis (P=exp(-8/T)). Pero, ¿qué ocurre con los spines que están en dicha frontera? Pues hemos de notar que estos spines, puesto que solo interactúan con sus cuatro vecinos más próximos, tan solo perciben que uno de sus vecinos (el único que pertenece a la otra fase) se encuentra alineado antiparalelamente a él, por lo que el incremento de energía en la propuesta de cambio de spin es bastante desfavorable, en el sentido de que dicho cambio será altamente improbable. En concreto, se tendrá que P=exp(-4/T), con T baja (p.e. T=0.05), lo cual hace P\simeq 0. Fenomenológicamente, no solo habremos capturado el fenómeno de la magnetización a temperaturas bajas, sino que también habremos capturado el fenómeno de la existencia de distintas fases antiparalelamente magnetizadas, pero hemos de notar que esto no es favorable si queremos reproducir las curvas de magnetización con M->1 cuando T->0, pues en ese caso de fases antiparalelamente magnetizadas se tienen dos grandes componentes con un valor macroscópico de magnetización y de distinto signo, los cuales resultan en M<<1 para T->0. Quizás, con un tiempo de termalización grandísimo, conseguiríamos que de ambas grandes fases se alineasen, pues después de todo P=exp(-4/T)>exp(-8/T)\simeq 0, pero esto no nos conviene en lo que a tiempo de computación se refiere. Por esto, inicializando el sistema uniformemente, evitamos este problema. Aunque, en ese caso, cabe preguntarse: ¿no tendremos problemas de un tiempo de termalización grande con inicialización uniforme cuando T sea grande? No, porque a T grande se pretende maximizar la entropía, i.e. el desorden, luego partiendo de una configuración uniforme a T grande, es bastante probable que aceptemos casi cualquier cambio que propongamos.

int main(void)
{
	//DO YOU NEED TO CHANGE MAX_SIDE AS WELL??
	const int side=60;
	const int howManyTs=100;
	const double T_ini = 1;
	const double T_step = 0.025;
	const int term_pMC = 1000;
	const int delta_pMC = 100;
	const int n_measurements = 10000;
	const string outPath = "data.dat";
	const double kB = 1.38064852e-23;
	const int snitch_every = 1;
	const bool verbose = true;
	
	int N=side*side;
	int i, ij;
	int neighbours_array[4][max_side*max_side], system[max_side*max_side];
	double T_array[howManyTs], minimums_array[5], mag, mag2, mag4, aux;
	ofstream output_file;
	
	gsl_rng * r; //Global random generator
	r = gsl_rng_alloc(gsl_rng_mt19937);
	gsl_rng_set(r, time(NULL));
	
	output_file.open(outPath);
	output_file << "#Side of spin grid: " << side << endl;
	output_file << "#Number of different temperatures: " << howManyTs << endl;
	output_file << "#Lowest temperature: " << T_ini << endl;
	output_file << "#Temperature resolution: " << T_step << endl;
	output_file << "#Number of termalization montecarlo steps: " << term_pMC << endl;
	output_file << "#Number of montecarlo steps between consecutive measurements: " << delta_pMC << endl;
	output_file << "#Number of measurements of each physical magnitudes: " << n_measurements << endl;
	output_file << "#T	mag	sigma(mag)	susc. mag	U4" << endl;

	//The neighbours array only depend on the size of the spin grid. Therefore, we can compute it outside the temperatures loop.
	//See above why we initialize the system uniformly
	initialize_uniform_system(r, system, side);
	build_neighbours_array(side, neighbours_array);
	build_T_array(howManyTs, T_ini, T_step, T_array);
	for(i=0; i<howManyTs; i++)
	{	
		//The minimums array depend on the temperature. We compute it here, inside the temperatures loop where the value of T is already fixed.
		build_minimums_array(T_array[i], minimums_array);
		//Let the system thermalize.
		evolve_N_pMC(term_pMC, r, side, minimums_array, neighbours_array, system);
		
		//Code for initializing measures goes here
		mag = 0.0;
		mag2 = 0.0;
		mag4 = 0.0;
		
		for(ij=0; ij<n_measurements; ij++)
		{	
			evolve_N_pMC(delta_pMC, r, side, minimums_array, neighbours_array, system);
			
			//Code for measuring and store its running sum goes here
			aux = measure_magnetization(system, side);
			mag = mag + aux;
			mag2 = mag2 + pow(aux,2.0);
			mag4 = mag4 + pow(aux,4.0);
			
		}
		
		//Code for normalizing the measurements made
		mag = mag/double(n_measurements);
		mag2 = mag2/double(n_measurements);
		mag4 = mag4/double(n_measurements);

		aux = (mag2-pow(mag,2.0));
		output_file << T_array[i] << "	" << mag << "	" << (n_measurements/(n_measurements-1))*aux << "	";
		aux = (double(N)/T_array[i])*aux;
		//We are writting susc_mag*kB
		output_file << aux << "	";
		aux = 1.0-(mag4/(3.0*pow(mag2,2.0)));
		output_file << aux << endl;
		
		if(verbose)
		{
			if(i%snitch_every==0)
			{
				cout << "Progress: " << 100*double(i)/double(howManyTs) << "%" << endl;
			}
		}
	}
	output_file.close();
	return 0;
}

//--------------FUNCTIONS--------------//

//Computes python-like a%b. b must be a positive integer (b>0). This function returns a%b if a>0, whereas it returns (b+a)%b if a<0.
int remainder(int a, int b)
{
	if(b<1)
	{
		cout << "remainder(), Err1" << endl;
		return -1;
	}
	if(a<0)
	{
		a = b+a;
	}
	return a%b;
}

//Gets the index of the k-th spin in the unidimensional array. It also gets two integer indices, i and j, by reference. The function writes the (i,j) bidimensional coordinates of such spin into i and j indices. Both the direct_translator and the inverse_translator are meant to be used with the C++ indexing system, i.e. starting in 0, so that the bidimensional grid is indexed from [0,0] through [side-1,side-1].
void direct_translator(int &i, int &j, int k, int side)
{
//	The division of two integer numbers is converted into an integer (in C) by truncation of the decimal part which is, in the end, the floor operation. Therefore, it is not necessary to need such flooring here.
	i = k/side;
	j = k%side;
	return;
}	

//The same translation process as in direct_translator(), but the other way around.
void inverse_translator(int i, int j, int &k, int side)
{
	k = (side*i)+j;
	return;
}

//neighbours[][] must have been defined beforehand to have side*side length along its 0-th axis, i.e. it is indexed from 0 to (side*side)-1. This function has been CHECKED TO WORK AS EXPECTED for side=5 and side=6.
void build_neighbours_array(int side, int neighbours[][max_side*max_side])
{
	int i, j, k;
	for(k=0; k<(side*side); k++)
	{
		direct_translator(i,j,k,side);
		//Periodic toroidal conditions are implicitly taken here by using the remainder function
		inverse_translator(remainder(i+1,side),j,neighbours[0][k],side);
		inverse_translator(i,remainder(j+1,side),neighbours[1][k],side);
		inverse_translator(remainder(i-1,side),j,neighbours[2][k],side);
		inverse_translator(i,remainder(j-1,side),neighbours[3][k],side);
	}
	return;
}

//This function has been checked to work properly
void build_T_array(int howManyTs, double T_ini, double T_step, double T_array[])
{
	int i;
	for(i=0; i<howManyTs; i++)
	{
		T_array[i] = T_ini +(i*T_step);
	}
	return;
}

void initialize_random_system(gsl_rng* r, int system[], int side)
{
	int i;
	for(i=0; i<(side*side); i++)
	{
		if(gsl_rng_uniform(r)<=0.5)
		{
			system[i]=1;
		}
		else
		{
			system[i] =-1;
		}
	}
	return;
}

void initialize_uniform_system(gsl_rng* r, int system[], int side)
{
	int i;
	if(gsl_rng_uniform(r)<=0.5)
	{
		for(i=0; i<(side*side); i++)
		{
			system[i] = +1;
		}
	}
	else
	{
		for(i=0; i<(side*side); i++)
		{
			system[i] = -1;
		}
	}
	return;
}

//This function maps the indices (0,1,2,3,4) to delta_E (-8,-4,0,4,8) respectively.
int index_to_delta_E(int index)
{
	return (4*(index-2));
}

//This function maps the delta_E (-8,-4,0,4,8) to indices (0,1,2,3,4) respectively.
int delta_E_to_index(int delta_E)
{
	//Note that delta_E must be divisible by 4
	return ((delta_E/4)+2);
}

//Generate array of minimums
void build_minimums_array(double T, double minimums_array[])
{
	int i;
	for(i=0; i<5; i++)
	{
		minimums_array[i] = min(1.0,exp(-1.0*double(index_to_delta_E(i))/T));
	}
	return;
}

//narr[][] is the neighbours array which must have been built according to build_neighbours_array().
void evolve_one_pMC(gsl_rng* r, int side, double minimums_array[], int narr[][max_side*max_side], int system[])
{
	int i, chosen_spin, aux;
	for(i=0; i<(side*side); i++)
	{
		chosen_spin = gsl_rng_uniform_int(r, side*side);
		aux = 2*system[chosen_spin]*(system[narr[0][chosen_spin]]+system[narr[1][chosen_spin]]+system[narr[2][chosen_spin]]+system[narr[3][chosen_spin]]);
		//Up to here, aux must hold some integer value between {-8,-4,0,4,8}
		aux = delta_E_to_index(aux);
		//Now aux must hold some integer value in {0,1,2,3,4}
		if(gsl_rng_uniform(r)<minimums_array[aux])
		{
			system[chosen_spin] = -1*system[chosen_spin];
		}
	}
	return;	
}

void evolve_N_pMC(int N, gsl_rng* r, int side, double minimums_array[], int narr[][max_side*max_side], int system[])
{
	int i;
	for(i=0; i<N; i++)
	{
		evolve_one_pMC(r, side, minimums_array, narr, system);
	}
	return;
}

//This function returns the magnetization of the flattened spin grid system.
double measure_magnetization(int system[], int side)
{
	int i;
	double magnetization = 0.0;
	for(i=0; i<(side*side); i++)
	{
		magnetization = magnetization+double(system[i]);
	}
	return abs(magnetization)/(1.0*side*side);
}
		
	
void normalize_measurements_array(double measurements[], int length, int n_measurements)
{
	int i;
	for(i=0; i<length; i++)
	{
		measurements[i] = measurements[i]/double(n_measurements);
	}
	return;
}

void print_grid(int system[], int side)
{
	int i;
	cout << "____________________________" << endl;
	for(i=0; i<(side*side); i++)
	{
		if(i%side==0)
		{
			cout << endl << i/side << "	";
		}
		cout << system[i] << "	";
	}
	cout << endl << "____________________________" << endl;
	return;
}

	
	




			
			
