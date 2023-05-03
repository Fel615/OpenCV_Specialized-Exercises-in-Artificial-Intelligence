#define _CRT_SECURE_NO_DEPRECATE
#include <Leap.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <stdio.h>
#include <string>
#include <cstdlib>  

# include <opencv2/highgui.hpp>
# include <opencv2/videoio.hpp>
# include <iostream>

#include <windows.h>
#include <time.h>     // for clock()

#pragma comment(lib,"Winmm.lib")//この行の代わりに「追加の依存ファイル」に追加しても良い
#include <mmsystem.h>

#define PLAY_SPEED 1.0
#define ESCAPE_KEY 27

using namespace cv;
using namespace std;
using namespace Leap;



int movie(string file_name,int bgm) {
	VideoCapture cap(file_name);//ファイルを開いてcap（動画）を読み込む
	if (!cap.isOpened()) return -1;//開けなかったら終了
	int max_frame = cap.get(cv::CAP_PROP_FRAME_COUNT);
	int max_fps = cap.get(cv::CAP_PROP_FPS);
	clock_t start = clock();    // スタート時間

	CascadeClassifier cascade;
	cascade.load("haarcascade_frontalface_alt.xml"); //正面顔情報が入っているカスケードファイル読み込み
	Mat fg = imread("catura.png", IMREAD_UNCHANGED), fgresize;//かつら

	Mat img;
	if (bgm == 0) {
		PlaySound(TEXT("Paper_win.wav"), NULL, SND_FILENAME | SND_ASYNC);
	}
	else if (bgm == 1) {
		PlaySound(TEXT("Paper_lose.wav"), NULL, SND_FILENAME | SND_ASYNC);
	}
	else if (bgm == 2) {
		PlaySound(TEXT("Scissors_win.wav"), NULL, SND_FILENAME | SND_ASYNC);
	}
	else if (bgm == 3) {
		PlaySound(TEXT("Scissors_lose.wav"), NULL, SND_FILENAME | SND_ASYNC);
	}
	else if (bgm == 4) {
		PlaySound(TEXT("brock_lose.wav"), NULL, SND_FILENAME | SND_ASYNC);
	}
	else if (bgm == 5) {
		PlaySound(TEXT("brock_lose.wav"), NULL, SND_FILENAME | SND_ASYNC);
	}





	for (int CFN = 0; CFN < max_frame; CFN++) {
		cap >> img; //1フレーム分取り出してimgに保持させる







		cvtColor(img, img, COLOR_RGB2RGBA);//チャンネル数を4に合わせる


		vector<cv::Rect> faces;
		cascade.detectMultiScale(img, faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, Size(20, 20));

		for (int i = 0; i < faces.size(); i++) //検出した顔の個数"faces.size()"分ループを行う
		{

			double rate = (double)faces[i].width / fg.cols;
			double rate2 = (double)faces[i].width / fg.cols * 4 / 5;
			resize(fg, fgresize, Size(), rate, rate2);//fgは猫耳
			if (faces[i].y >= 100) {
				Mat roi = img(Rect(faces[i].x, 0, fgresize.cols, fgresize.rows));
				//左基準点をすこし上に上げると猫耳が上に上がる　 faces[i].y→ faces[i].y-50など
				fgresize.copyTo(roi, fgresize);//fgresize(かつらのリサイズされたもの)をroiに貼り付け
			}

		}
		//Sleep(20);    画像処理で何か重たい処理をする場合, ここでは20ms画像処理に時間がかかるとする
		//img = myEffect(img);  のように、ここで表示する動画にリアルタイムでエフェクトをかける
		cv::imshow("screen", img); //画像を表示




		clock_t end = clock();
		int wait_time = int((double)CFN / (30.0 * PLAY_SPEED) * 1000.0 - (double)(end - start));

		//追加部分
		if (wait_time < -40) {
			int loop_num = -wait_time / 33.3;  //送れているコマ数を計算（FPS=30の動画の場合、一コマ当たり33.3ms）
			for (int i = 0; i < loop_num; i++) {
				cap >> img;  //現在の表示フレームが遅れている場合、その分だけフレームナンバーを進める
				CFN++;
			}
			std::cout << loop_num << "\n";
		}

		int key = cv::waitKey(int(min(max(wait_time, 1), 50))); // 表示のために1ms待つ

		if (key == ESCAPE_KEY) {
			// esc or enterキーで終了
			cv::destroyAllWindows();

		}
	}
	PlaySound(NULL, NULL, 0);
	cv::destroyWindow("screen");


	exit(1);
}

int movie_first(string file_name) {
	VideoCapture cap(file_name);//ファイルを開いてcap（動画）を読み込む
	if (!cap.isOpened()) return -1;//開けなかったら終了
	int max_frame = cap.get(cv::CAP_PROP_FRAME_COUNT);
	int max_fps = cap.get(cv::CAP_PROP_FPS);
	clock_t start = clock();    // スタート時間
	CascadeClassifier cascade;
	cascade.load("haarcascade_frontalface_alt.xml"); //正面顔情報が入っているカスケードファイル読み込み
	Mat fg = imread("catura.png", IMREAD_UNCHANGED), fgresize;//かつら
	Mat img;
	PlaySound(TEXT("intro.wav"), NULL, SND_FILENAME | SND_ASYNC);


	for (int CFN = 0; CFN < max_frame; CFN++) {
		cap >> img; //1フレーム分取り出してimgに保持させる

		cvtColor(img, img, COLOR_RGB2RGBA);//チャンネル数を4に合わせる


		vector<cv::Rect> faces;
		cascade.detectMultiScale(img, faces, 1.1, 3, 0 | cv::CASCADE_SCALE_IMAGE, Size(20, 20));

		for (int i = 0; i < faces.size(); i++) //検出した顔の個数"faces.size()"分ループを行う
		{

			double rate = (double)faces[i].width / fg.cols;
			double rate2 = (double)faces[i].width / fg.cols*4/5;
			resize(fg, fgresize, Size(), rate, rate2);//fgは猫耳
			if (faces[i].y>=100) {
				Mat roi = img(Rect(faces[i].x, 0, fgresize.cols, fgresize.rows));
				//左基準点をすこし上に上げると猫耳が上に上がる　 faces[i].y→ faces[i].y-50など
				fgresize.copyTo(roi, fgresize);//fgresize(かつらのリサイズされたもの)をroiに貼り付け
			}

		}






		//Sleep(20);    画像処理で何か重たい処理をする場合, ここでは20ms画像処理に時間がかかるとする
		//img = myEffect(img);  のように、ここで表示する動画にリアルタイムでエフェクトをかける
		cv::imshow("screen", img); //画像を表示

		clock_t end = clock();
		int wait_time = int((double)CFN / (30.0 * PLAY_SPEED) * 1000.0 - (double)(end - start));

		//追加部分
		if (wait_time < -40) {
			int loop_num = -wait_time / 33.3;  //送れているコマ数を計算（FPS=30の動画の場合、一コマ当たり33.3ms）
			for (int i = 0; i < loop_num; i++) {
				cap >> img;  //現在の表示フレームが遅れている場合、その分だけフレームナンバーを進める
				CFN++;
			}
			std::cout << loop_num << "\n";
		}

		int key = cv::waitKey(int(min(max(wait_time, 1), 50))); // 表示のために1ms待つ

		if (key == ESCAPE_KEY) {
			// esc or enterキーで終了
			cv::destroyAllWindows();
			
		}
	}
	PlaySound(NULL, NULL, 0);
	cv::destroyWindow("screen");


}



int main(int argc, char** argv) {
	int width = 780;
	int height = 700;
	char Number[5];
	Controller controller;
	int extendedFingers = 0, nowfinger = 0;
	double multi = 1.4;
	int x_offset = 200;
	int ransu=150;//ransu分の1の確立で勝てる
	int resl = 0;
	int resr = 0;


	string filename = "intro.mp4";//開くファイル
	movie_first(filename);
	
	
	while (1) {

		Mat draw_zone(Size(width, height), CV_8UC3, Scalar::all(255));
		Leap::Frame frame = controller.frame(); // controller is a Leap::Controller object
		Leap::HandList hands = frame.hands();
		Leap::FingerList fingers = frame.fingers();
		Hand hand = frame.hands()[0]; //0
		extendedFingers = 0; //指本数を数える
		
		for (int f = 0; f < frame.fingers().count(); f++) {
			Finger finger = frame.fingers()[f];
			if (finger.isExtended()) extendedFingers++;
		}
		//現在の本数 real time表示
		sprintf(Number, "Real time Number=%d", extendedFingers);
		cv::putText(draw_zone, Number, cv::Point(50, 100), cv::FONT_HERSHEY_SIMPLEX, 1.4, cv::Scalar(255, 0, 0), 5);
		if (extendedFingers == 5) {
			printf("あなたが出したのはパーです");
			/*sprintf(Number, "Paper");
			cv::putText(draw_zone, Number, cv::Point(50, 150), cv::FONT_HERSHEY_SIMPLEX, 1.4, cv::Scalar(255, 128, 0), 5);*/
			int v1 = rand() % ransu;
			if (v1 == 0) {
				string filename = "Paper_win.mp4";
				destroyWindow("drawzone");
				movie(filename,0);

			}
			else{
				string filename = "Paper_lose.mp4";
				destroyWindow("drawzone");
				movie(filename, 1);


			}
			
		}
		if (extendedFingers == 2) {
			for (int f = 0; f < frame.fingers().count(); f++) {
				Finger finger = frame.fingers()[f];
				if (finger.isExtended()) {
					if (f == 1) {
						resl = 1;
					}
					if (f == 2) {
						resr = 1;
					}
				}
			}
			if (resl == 1) {
				if (resr == 1) {
					printf("あなたが出したのはチョキです");
					int v1 = rand() % ransu;
					if (v1 == 0) {
						string filename = "Scissors_win.mp4";
						destroyWindow("drawzone");
						movie(filename, 2);
					}
					else {
						string filename = "Scissors_lose.mp4";
						destroyWindow("drawzone");
						movie(filename, 3);
					}
				}
			}

		}
		if (extendedFingers == 0) {
			printf("あなたが出したのはグーです");
			int v1 = rand() % ransu;
			if (v1 == 0) {
				string filename = "brock_win.mp4";
				destroyWindow("drawzone");
				movie(filename,4);

			}
			else {
				string filename = "brock_lose.mp4";
				destroyWindow("drawzone");
				movie(filename, 5);

			}


			/*sprintf(Number, "Rock");
			cv::putText(draw_zone, Number, cv::Point(50, 150), cv::FONT_HERSHEY_SIMPLEX, 1.4, cv::Scalar(255, 128, 0), 5);*/
		}

		/*for (auto finger : frame.fingers().extended()) {
			std::cout << "ID : " << finger.id()
				<< " 種類 : " << finger.type()
				<< " 位置 : " << finger.tipPosition()
				<< " 速度 : " << finger.tipVelocity()
				<< " 向き : " << finger.direction()
				<< std::endl;
		}*/

		for (int i = 0; i < 5; i++) {
			circle(draw_zone, Point(hand.palmPosition().x + x_offset, 700 - hand.palmPosition().y), 45, Scalar(0, 0, 0), -1);
			circle(draw_zone, Point(fingers[i].tipPosition().x * multi + x_offset, 700 - fingers[i].tipPosition().y * multi), 15, Scalar(0, 0, i * 50 + 50), -1);
			line(draw_zone, Point(fingers[i].tipPosition().x * multi + x_offset, 700 - fingers[i].tipPosition().y * multi),
				Point(hand.palmPosition().x + x_offset, 700 - hand.palmPosition().y), Scalar(255, 255, 0), 5, 8); //親指中心と結ぶ
		}

		imshow("drawzone", draw_zone);
		waitKey(30);
		

	}
	//leapdata.close();
	
	
}