import cv2
import numpy as np
import random
from copy import copy

class PNGOverlay():
    def __init__(self, filename):
        # アルファチャンネル付き画像(BGRA)として読み込む
        self.src_init = cv2.imread(filename, -1)

        #必要最低限の透明色画像を周囲に付加する
        self.src_init = self._addTransparentImage(self.src_init)

        #画像の変形はデフォルトは不要
        self.flag_transformImage = False

        #画像の前処理を行う
        self._preProcessingImage(self.src_init)

        #初期値
        self.degree = 0
        self.size_value = 1

    def _addTransparentImage(self, src): #回転時にクロップしないように、予め画像の透明色領域を周囲に加える
        height, width, _ = src.shape # HWCの取得

        #回転対応で、対角線の長さを一辺とする透明色の正方形を作る
        diagonal = int(np.sqrt(width **2 + height ** 2))
        src_diagonal = np.zeros((diagonal, diagonal, 4), dtype=np.uint8)

        #透明色の正方形の中心にsrcを上書き
        p1 = int(diagonal/2 - width/2)
        p2 = p1 + width
        q1 = int(diagonal/2 - height/2)
        q2 = q1 + height
        src_diagonal[q1:q2,p1:p2,:] = src[:,:,:]

        return src_diagonal

    def _preProcessingImage(self, src_bgra): #BGRA画像をBGR画像(src)とA画像(mask)に分け、オーバーレイ時に必要な情報を保持
        self.mask = src_bgra[:,:,3]  # srcからAだけ抜き出し mask とする
        self.src = src_bgra[:,:,:3]  # srcからGBRだけ抜き出し src とする
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)  # Aを3チャンネル化
        self.mask = self.mask / 255.0  # 0.0-1.0に正規化
        self.height, self.width, _ = src_bgra.shape # HWCの取得
        self.flag_preProcessingImage = False #前処理フラグは一旦下げる

    def rotate(self, degree): #画像の回転パラメータ受付
        self.degree = degree
        self.flag_transformImage = True

    def resize(self, size_value): #画像のサイズパラメータ受付
        self.size_value = size_value
        self.flag_transformImage = True

    def _transformImage(self): #各メソッドでバラバラに行わず、一括でリサイズと回転をシリーズに行う必要がある
        #---------------------------------
        #resize
        #---------------------------------
        self.src_bgra = cv2.resize(self.src_init, dsize=None, fx=self.size_value, fy=self.size_value) #倍率で指定

        #サイズを変えたのでwidthとheightを出し直す
        self.height, self.width, _ = self.src_bgra.shape # HWCの取得

        #---------------------------------
        #rotate
        #---------------------------------
        #getRotationMatrix2D関数を使用
        center = (int(self.width/2), int(self.height/2))
        trans = cv2.getRotationMatrix2D(center, self.degree, 1)

        #アフィン変換
        self.src_bgra = cv2.warpAffine(self.src_bgra, trans, (self.width, self.height))

        #変形は終了したのでフラグはFalseにする
        self.flag_transformImage == False

        #オーバーレイの前に画像の前処理を行う
        self.flag_preProcessingImage = True

    def show(self, dst, x, y): #dst画像にsrcを重ね合わせ表示。中心座標指定
        #回転とサイズ変更はoverlayの直前で一括して行う必要がある
        if self.flag_transformImage == True:
            self._transformImage()

        #前処理が必要な場合は実行
        if self.flag_preProcessingImage == True:
            self._preProcessingImage(self.src_bgra)

        x1, y1 = x - int(self.width/2), y - int(self.height/2)
        x2, y2 = x1 + self.width, y1 + self.height #widthやheightを加える計算式にしないと１ずれてエラーになる場合があるので注意
        a1, b1 = 0, 0
        a2, b2 = self.width, self.height
        dst_height, dst_width, _ = dst.shape

        #x,y指定座標が dstから完全にはみ出ている場合は表示しない
        if x2 <= 0 or x1 >= dst_width or y2 <= 0 or y1 >= dst_height:
            return

        #dstのフレームからのはみ出しを補正
        x1, y1, x2, y2, a1, b1, a2, b2 = self._correctionOutofImage(dst, x1, y1, x2, y2, a1, b1, a2, b2)

        # Aの割合だけ src 画像を dst にブレンド
        dst[y1:y2, x1:x2] = self.src[b1:b2, a1:a2] * self.mask[b1:b2, a1:a2] + dst[y1:y2, x1:x2] * ( 1 - self.mask[b1:b2, a1:a2] )

    def _correctionOutofImage(self, dst, x1, y1, x2, y2, a1, b1, a2, b2): #x, y座標がフレーム外にある場合、x, y及びa, bを補正する
        dst_height, dst_width, _ = dst.shape
        if x1 < 0:
            a1 = -x1
            x1 = 0
        if x2 > dst_width:
            a2 = self.width - x2 + dst_width
            x2 = dst_width
        if y1 < 0:
            b1 = -y1
            y1 = 0
        if y2 > dst_height:
            b2 = self.height - y2 + dst_height
            y2 = dst_height

        return x1, y1, x2, y2, a1, b1, a2, b2


def rand_nodup(a, b, k):
    if abs(a) + b < k:
        raise ValueError
    r = set()
    while len(r) < k:
        r.add(random.randint(a, b))
    return list(r)


def face_scale(src):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(src_gray)
    for x, y, w, h in faces:
        cv2.rectangle(src, (x, y), (x + w, y + h), (255, 255, 0), 2)
    return src



def hconcat(img_list):


    #最も小さい高さを収得する
    min_height = 200

    #最も小さい高さに合わせてサイズの変更
    img_list_resize = [cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height))
                      for img in img_list]
    return cv2.hconcat(img_list_resize)

def vconcat(img_list):

    #最も小さい幅を収得する
    min_width = min(img.shape[1] for img in img_list)

    #最も小さい幅に合わせてサイズの変更
    img_list_resize = [cv2.resize(img, (min_width, int(img.shape[0] * min_width / img.shape[1])))
                      for img in img_list]
    return cv2.vconcat(img_list_resize)

def stack_create(ranlst,sc):
    im_st=[]
    im_st_base=[]
    l=1
    for i,num in enumerate(ranlst):
        src = cv2.imread(f'face_{num}.png')
        src = cv2.resize(src, dsize=(200, 200))

        #src=face_scale(src)
        cv2.putText(src,
                    text=f"{i+1}",
                    org=(0,30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_4)

        im_st_base.append(src)

        if l%sc==0:
            im_st.append(im_st_base)
            im_st_base=[]
        if l==(len(ranlst)) and len(im_st_base)>0:
            #print(len(im_st_base))
            bl=np.zeros((200, 200, 3))
            bl= cv2.resize(bl, dsize=(200, 200))
            bl=bl.astype('uint8')
            for i in range(sc-len(im_st_base)):
                im_st_base.append(bl)
            im_st.append(im_st_base)
            break
        elif l==(len(ranlst)):
            
            break
        l+=1
    im_st_base=[]
    #print("fin",len(im_st))
    for i,name in enumerate(im_st):
        im_st_base.append(hconcat(name))
    return vconcat(im_st_base)


def feature_matching(filename1,filename2):

    # 白黒画像で画像を読み込み
    img1 = filename1
    img2 = filename2

    # ORB (Oriented FAST and Rotated BRIEF)
    detector = cv2.AKAZE_create()

    # 特徴検出
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # マッチング処理
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # 特徴記述子のマッチング
    matches = bf.match(des1, des2)
    # 距離でソートする
    matches = sorted(matches, key = lambda x:x.distance)
    
    

    # マッチング上位20個の特徴点を線でリンクして画像に書き込む
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)

    return match_img



   

def check_the_answer():
    for ratio in [round(num * 0.01, 2) for num in range(1, 61,5)]:
            src= cv2.imread(filename_qes)
            src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(src_gray)
            for x, y, w, h in faces:
                small = cv2.resize(src[y: y + h, x: x + w], None, fx=ratio, fy=ratio)
                src[y: y + h, x: x + w] = cv2.resize(small, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        
            
            mihon=src[y: y + h, x: x + w]
            ans=feature_matching(mihon,img_stack_ans)
            cv2.imshow("model",ans)
            cv2.waitKey(500)
    cv2.waitKey(0)
    return


def start(sentaku):
    print(f"""　||￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣￣||
　|| ★★私は誰でしょう？★★                                     
　||　　　　～　たくさんの画像から見つけ出せ！　～             
　||                                                              
　|| ●モザイクのかかった画像が表示されるから誰のものか当ててね 
　||　一発で当てたらすごい！                                      
　||                                                              
　|| ▲easy normal hardの三つがあるよ！
　||　easyがやりたければ1、normalがやりたければ2、hardがやりたければ3を押してEnter
  ||
  || ◎正解したら答え合わせが待ってるよ
  ||　まぁ解答閉め切ってもでるけどね
　||
　|| ■準備が出来たらEnterをクリック
　||　                                ｡ 　　∧_∧
　||　　　　　　　　　　　　　　　　　 ＼　(ﾟーﾟ*)　ｷﾎﾝ。
　||＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿＿ 　     ⊂⊂ |
　　　　                                |￣￣￣￣|
                                        |　　　　|
                                         ￣￣￣

""")
    while True:
       key = input('難易度の数字を入力して Enterキーを押したらゲームを開始します')
       if key=="1" :
           kaku = input('easyで選択肢は5、チャンスは4回です。問題なければ Enterキーを押したらゲームを開始します。再選択したいなら0を入力してください。')
           if not kaku:
               return 5,5,4
           elif kaku=="0":
               continue
       elif key=="2":
           kaku = input('normalで選択肢は8、チャンスは3回です。問題なければ Enterキーを押したらゲームを開始します。再選択したいなら0を入力してください。')
           if not kaku:
               return 8,4,3
           elif kaku=="0":
               continue
       elif key=="3":
           kaku = input('hardで選択肢は12、チャンスは3回です。問題なければ Enterキーを押したらゲームを開始します。再選択したいなら0を入力してください。')
           if not kaku:
               return 12,4,3
           elif kaku=="0":
               continue
       elif key=="4":
           kaku = input('lunaticの存在にきづくとは、選択肢は最大の17、チャンスはたったの2回です。問題なければ Enterキーを押したらゲームを開始します。再選択したいなら0を入力してください。')
           if not kaku:
               return 17,6,2
           elif kaku=="0":
               continue 
       elif not key:
           continue



def main(sentaku,ans_res,ratio,finish):

    #for文で倍率を上げていく
    for ratio in [round(num * 0.01, 2) for num in range(1, 51,1)]:
        src= cv2.imread(filename_qes)
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)


        faces = face_cascade.detectMultiScale(src_gray)
        for x, y, w, h in faces:
            small = cv2.resize(src[y: y + h, x: x + w], None, fx=ratio, fy=ratio)
            src[y: y + h, x: x + w] = cv2.resize(small, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        
        
        #問題を出題
        question=src[y: y + h, x: x + w]
        cv2.imshow("Who is this image?",question)
        print("Q:Who is this image?")
        cv2.imshow("choose",img_stack)
        cv2.waitKey(100)
            

        #答えを入力
        ans=input("M:正解は左上に対応する番号を半角数字でお答えください>")
        ans=int(ans)
        cv2.destroyAllWindows()
            

        #カンニング用
        #print(f"model{int(model_answer)+1},ans{ans}")

        #解答回数を増加
        ans_res+=1

        #模範解答と答えが同じなら答え合わせの関数に移行し終了
        if int(ans)==(model_answer+1):
            print("A:正解！！おめでとうございます！")
            check_the_answer()
            finish=True

        #模範解答と答えが不一致なら間違えた選択肢に×をつける
        else:
            print(f"A:不正解！チャンスはあと{res_max-ans_res}回！レートを上げたのでもう一度挑戦しよう")
            ans_gyo=(ans-1)//que_retu
            ans_retu=(ans-1)%que_retu
            item.show(img_stack, 100+(200*(ans_retu)), 100+(200*(ans_gyo)))

        #残り二択で間違えたら終了
        if ans_res==(res_max):
                check_the_answer()
                finish=True

        #終了処理
        if finish==True:
            return
      



if __name__ == "__main__":
    #選択肢の数
    sentaku=17

    #改行の個数
    que_retu=6

    res_max=5

    #sentaku,que_retu,res_max=start(sentaku)

    

    #カスケードファイルの設定
    face_cascade_path = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    #選択肢の画像を五枚選ぶ
    ran=rand_nodup(1, 17, sentaku)

    #選択肢の中から答えを選ぶ
    model_answer=random.randint(0,sentaku-1)

    #使う画像を読み込み
    filename_qes = f'face_{ran[model_answer]}.png'
    img_stack=stack_create(ran,que_retu)
    img_stack_ans=copy(img_stack)
    item = PNGOverlay("batu.png")
    item.resize(0.1) 

    #解答回数の初期化
    ans_res=0

    #モザイクの倍率を決定
    ratio = 0.01

    #終了条件の初期化
    finish=False

    
    main(sentaku,ans_res,ratio,finish)
        
cv2.destroyAllWindows()