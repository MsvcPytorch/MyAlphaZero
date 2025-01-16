#include<bits/stdc++.h>
#include<Windows.h>
using namespace std;
signed main(){
	int cnt=0;
	while(1){
		cnt++;
		cerr<<"Gen "<<cnt<<" Started"<<endl;
		cerr<<"Self Playing..."<<endl;
		system("selfPlay.bat");
	//	Sleep(30*60*1000);
	//	Sleep(20*1000);
		cerr<<"Training..."<<endl;
		system("startTrain.bat");
	//	Sleep(6*60*1000);
	}
	return 0;
}
