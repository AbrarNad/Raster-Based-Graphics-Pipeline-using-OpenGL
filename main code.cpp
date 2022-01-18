#include<cmath>
#include<bits/stdc++.h>
#include<iostream>
#include<fstream>
#include <time.h>
#include "bitmap_image.hpp"

#define N 4
#define pi (2*acos(0.0))

using namespace std;

struct point
{
    double x,y,z;
};

class matrix
{
public:
    double arr[4][4];

    matrix(double matrix[4][4])
    {
        for(int i=0; i<4; i++)
        {
            for(int j=0; j<4; j++)
            {
                arr[i][j] = matrix[i][j];
            }
        }
    }

    matrix()
    {
    }
};

class triangle
{
public:
    struct point a,b,c;
    int color[3];

    triangle(struct point a,struct point b,struct point c)
    {
        this->a = a;
        this->b = b;
        this->c = c;

        for(int i=0; i<3; i++){
            color[i] = rand()%256;
        }
    }
};

struct point getUnitVect(struct point p){
    struct point ret;

    double s = sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
    ret.x = p.x/s;
    ret.y = p.y/s;
    ret.z = p.z/s;

    return ret;
};

double vectModulus(struct point p){

    double s = sqrt(p.x*p.x+p.y*p.y+p.z*p.z);

    return s;
};


struct point multVectScalar(struct point p, double a)
{
    struct point ret;

    ret.x=p.x*a;
    ret.y=p.y*a;
    ret.z=p.z*a;

    return ret;
};

struct point sumVect( point p, point q)
{
    struct point ret;

    ret.x=p.x+q.x;
    ret.y=p.y+q.y;
    ret.z=p.z+q.z;

    return ret;
};

struct point crossMult( point v, point w)
{
    struct point ret;

    ret.x = v.y*w.z - v.z*w.y;
    ret.y = v.z*w.x - v.x*w.z;
    ret.z = v.x*w.y - v.y*w.x;

    return ret;
};

double dotMult(point v, point w){
    double ret;
    ret = v.x*w.x+v.y*w.y+v.z*w.z;

    return ret;
};

struct point rotateAroundAxis( point v, point k, double a)
{
    struct point ret;

    ret = multVectScalar(v,cos(a*pi/180.0));
    ret = sumVect(ret,multVectScalar(crossMult(k,v),sin(a*pi/180.0)));

    return ret;
};

struct point Rotate(point x, point a, double angle){
    struct point ret,tmp;

    ret = multVectScalar(x,cos(angle*pi/180.0));
    ret = sumVect(ret,multVectScalar(crossMult(a,x),sin(angle*pi/180.0)));
    tmp = multVectScalar(a,1-cos(angle*pi/180.0));
    tmp = multVectScalar(tmp,dotMult(a,x));
    ret = sumVect(ret,tmp);

    return ret;
};

void printPoint(struct point p)
{
    cout<<endl<<endl;
    cout<<p.x<<" "<<p.y<<" "<<p.z<<endl;
}

string strPoint(struct point p){
    return to_string(p.x)+" "+to_string(p.y)+" "+to_string(p.z);
}

void printMatrix4(double arr[4][4]){
    cout<<endl;
    for(int i=0;i<4;i++){
        for(int j=0; j<4; j++){
            cout<<arr[i][j]<<" ";
        }
        cout<<endl;
    }
}

void matrixMult4v4(double mat1[4][4], double mat2[4][4], double res[4][4])
{
    int i,j,k;

    for (int h = 0; h < 4; h++)
    {

        for (int w = 0; w < 4; w++)
        {
            res[h][w] = 0;
        }
    }

    for (i = 0; i < 4; i++) {
        for (j = 0; j < 4; j++) {
            res[i][j] = 0;
            for (k = 0; k < 4; k++)
                res[i][j] += mat1[i][k] * mat2[k][j];
        }
    }

}

struct point matrixMult4v1(double mat1[4][4], double mat2[4][1])
{
    int i,j,k;
    struct point result;
    double sum;
    double res[4][1];


    for (i = 0; i < 4; i++) {
        sum = 0;
        for (j = 0; j < 4; j++) {
                sum += mat1[i][j] * mat2[j][0];
                res[i][0] = sum;
        }
    }

    result.x = res[0][0]/res[3][0];
    result.y = res[1][0]/res[3][0];
    result.z = res[2][0]/res[3][0];

    return result;
}

double getMaxY(triangle tr){
    return max({tr.a.y, tr.b.y, tr.c.y});
}

double getMinY(triangle tr){
    return min({tr.a.y, tr.b.y, tr.c.y});
}


double getMaxX(triangle tr){
    return max({tr.a.x, tr.b.x, tr.c.x});
}

double getMinX(triangle tr){
    return min({tr.a.x, tr.b.x, tr.c.x});
}

double getPointDist(struct point p, struct point q){
    return sqrt((p.x-q.x)*(p.x-q.x) + (p.y-q.y)*(p.y-q.y));
}

int main()
{
    srand(time(NULL));
    ///init top
    double identity[4][4]= {{1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1}};
    matrix top(identity);
    vector< matrix> Stack;
    vector< matrix> stateStack;     ///for push/pop
    vector< triangle> inpTriangles;
    vector< triangle> transTriangles;
    vector< triangle> transTrianglesS2;
    vector< triangle> transTrianglesS3;
    Stack.push_back(top);

    point eye, look, up;
    double fovY, aspectRatio, near, far;
    int screen_width, screen_height;
    double leftlimitX,rightlimitX,bottomlimitY,toplimitY,frontZ,rearZ;

    string filename = "scene.txt";

    fstream input;
    string line;

    input.open(filename, ios::in|ios::out|ios::app);
    int idx = 0;
    while(input)
    {
        getline(input,line);
        vector <string> tokens;
        stringstream check(line);
        string word;

        ///tokenizing input lines
        while(getline(check, word, ' '))
        {
            tokens.push_back(word);
        }

        if(idx ==0)
        {
            eye.x = stof(tokens[0]);
            eye.y = stof(tokens[1]);
            eye.z = stof(tokens[2]);
        }
        else if(idx ==1)
        {
            look.x = stof(tokens[0]);
            look.y = stof(tokens[1]);
            look.z = stof(tokens[2]);
        }
        else if(idx ==2)
        {
            up.x = stof(tokens[0]);
            up.y = stof(tokens[1]);
            up.z = stof(tokens[2]);
        }
        else if(idx ==3)
        {
            fovY = stof(tokens[0]);
            aspectRatio = stof(tokens[1]);
            near = stof(tokens[2]);
            far = stof(tokens[3]);
        }
        cout<<line<<endl;
        ///stage 2
        struct point l = sumVect(look,multVectScalar(eye,-1));
        l = getUnitVect(l);
        struct point r = crossMult(l,up);
        r = getUnitVect(r);
        struct point u = crossMult(r,l);
        double V[4][4];
        double T[4][4] = {{1,0,0,-eye.x},{0,1,0,-eye.y},{0,0,1,-eye.z},{0,0,0,1}};
        double R[4][4]= {{ r.x, r.y, r.z, 0},{ u.x, u.y, u.z, 0},{ -l.x, -l.y, -l.z, 0},{ 0, 0, 0, 1}};

        matrixMult4v4(R,T,V);

        ///stage 3
        double fovX = fovY*aspectRatio;
        double t = near*tan((fovY/2)*pi/180);
        double r2 = near*tan((fovX/2)*pi/180);
        //cout<<"stage 3"<<endl;
        //cout<<fovY<<" "<<fovX<<" "<<near<<" "<<far<<" "<<r2<<" "<<t<<" "<<tan((fovY/2)*pi/180)<<" "<<tan((fovX/2)*pi/180)<<endl;

        double P[4][4] = {{near/r2,0,0,0},{0,near/t,0,0},{0,0,-(far+near)/(far-near),-(2*far*near)/(far-near)},{0,0,-1,0}};

        ///stage 1 opts

        if(tokens.size()==1)
        {
            //cout<<"inside size 1"<<endl;
            cout<<tokens[0]<<endl;
            if(tokens[0] == "triangle")
            {
                cout<<"inside triangle"<<endl;
                struct point a,b,c;
                getline(input,line);
                cout<<line<<endl;
                vector<string> newTokens;
                stringstream check(line);
                string word;
                while(getline(check, word, ' '))
                {
                    newTokens.push_back(word);
                }

                a.x = stof(newTokens[0]);
                a.y = stof(newTokens[1]);
                a.z = stof(newTokens[2]);

                newTokens.clear();
                getline(input,line);
                cout<<line<<endl;
                stringstream check2(line);
                while(getline(check2, word, ' '))
                {
                    newTokens.push_back(word);
                }

                b.x = stof(newTokens[0]);
                b.y = stof(newTokens[1]);
                b.z = stof(newTokens[2]);

                newTokens.clear();
                getline(input,line);
                cout<<line<<endl;
                stringstream check3(line);
                while(getline(check3, word, ' '))
                {
                    newTokens.push_back(word);
                }

                c.x = stof(newTokens[0]);
                c.y = stof(newTokens[1]);
                c.z = stof(newTokens[2]);

                /*********/
                triangle tr(a,b,c);
                inpTriangles.push_back(tr);

                double matA[4][1] = {{a.x},{a.y},{a.z},{1}};
                double matB[4][1] = {{b.x},{b.y},{b.z},{1}};
                double matC[4][1] = {{c.x},{c.y},{c.z},{1}};

                struct point newA, newB, newC;

                newA = matrixMult4v1(top.arr,matA);
                newB = matrixMult4v1(top.arr,matB);
                newC = matrixMult4v1(top.arr,matC);

                triangle trans(newA,newB,newC);
                transTriangles.push_back(trans);

                ///stage 2
                double nmatA[4][1] = {{newA.x},{newA.y},{newA.z},{1}};
                double nmatB[4][1] = {{newB.x},{newB.y},{newB.z},{1}};
                double nmatC[4][1] = {{newC.x},{newC.y},{newC.z},{1}};

                newA = matrixMult4v1(V,nmatA);
                newB = matrixMult4v1(V,nmatB);
                newC = matrixMult4v1(V,nmatC);
                triangle stage2T(newA,newB,newC);
                transTrianglesS2.push_back(stage2T);

                ///stage 3
                double n3matA[4][1] = {{newA.x},{newA.y},{newA.z},{1}};
                double n3matB[4][1] = {{newB.x},{newB.y},{newB.z},{1}};
                double n3matC[4][1] = {{newC.x},{newC.y},{newC.z},{1}};

                newA = matrixMult4v1(P,n3matA);
                newB = matrixMult4v1(P,n3matB);
                newC = matrixMult4v1(P,n3matC);
                triangle stage3T(newA,newB,newC);
                transTrianglesS3.push_back(stage3T);

            }

            if(tokens[0] == "translate"){

                cout<<"inside translate"<<endl;
                struct point transCord;
                getline(input,line);
                cout<<line<<endl;
                vector<string> newTokens;
                stringstream check(line);
                string word;
                while(getline(check, word, ' '))
                {
                    newTokens.push_back(word);
                }

                transCord.x = stof(newTokens[0]);
                transCord.y = stof(newTokens[1]);
                transCord.z = stof(newTokens[2]);

                double matTrans[4][4] = {{1,0,0,transCord.x},{0,1,0,transCord.y},{0,0,1,transCord.z},{0,0,0,1}};
                double newTop[4][4];
                matrixMult4v4(top.arr,matTrans,newTop);

                cout<<"aager top"<<endl;
                printMatrix4(top.arr);
                cout<<"notun top"<<endl;
                printMatrix4(newTop);
                ///top.arr = newTop;
                for(int i=0;i<4;i++){
                    for(int j=0; j<4; j++){
                        top.arr[i][j] = newTop[i][j];
                    }
                }
                Stack.push_back(top);
            }

            if(tokens[0] == "scale"){
                cout<<"inside scale"<<endl;
                struct point scaleCord;
                getline(input,line);
                cout<<line<<endl;
                vector<string> newTokens;
                stringstream check(line);
                string word;
                while(getline(check, word, ' '))
                {
                    newTokens.push_back(word);
                }

                scaleCord.x = stof(newTokens[0]);
                scaleCord.y = stof(newTokens[1]);
                scaleCord.z = stof(newTokens[2]);

                double matScale[4][4] = {{scaleCord.x,0,0,0},{0,scaleCord.y,0,0},{0,0,scaleCord.z,0},{0,0,0,1}};
                double newTop[4][4];
                matrixMult4v4(top.arr,matScale,newTop);

                cout<<"aager top"<<endl;
                printMatrix4(top.arr);
                cout<<"notun top"<<endl;
                printMatrix4(newTop);
                ///top.arr = newTop;
                for(int i=0;i<4;i++){
                    for(int j=0; j<4; j++){
                        top.arr[i][j] = newTop[i][j];
                    }
                }
                Stack.push_back(top);
            }

            if(tokens[0] == "rotate"){
                cout<<"inside rotate"<<endl;
                struct point a;
                double angle;
                getline(input,line);
                cout<<line<<endl;
                vector<string> newTokens;
                stringstream check(line);
                string word;
                while(getline(check, word, ' '))
                {
                    newTokens.push_back(word);
                }

                angle = stof(newTokens[0]);
                a.x = stof(newTokens[1]);
                a.y = stof(newTokens[2]);
                a.z = stof(newTokens[3]);

                struct point c1,c2,c3;
                struct point x1,x2,x3;
                x1.x = 1;
                x1.y = 0;
                x1.z = 0;

                x2.x = 0;
                x2.y = 1;
                x2.z = 0;

                x3.x = 0;
                x3.y = 0;
                x3.z = 1;

                a = getUnitVect(a);
                c1 = Rotate(x1,a,angle);
                c2 = Rotate(x2,a,angle);
                c3 = Rotate(x3,a,angle);
                double matRot[4][4] = {{c1.x,c2.x,c3.x,0},{c1.y,c2.y,c3.y,0},{c1.z,c2.z,c3.z,0},{0,0,0,1}};
                double newTop[4][4];
                matrixMult4v4(top.arr,matRot,newTop);


                ///top.arr = newTop;
                for(int i=0;i<4;i++){
                    for(int j=0; j<4; j++){
                        top.arr[i][j] = newTop[i][j];
                    }
                }
                Stack.push_back(top);
            }

            if(tokens[0] == "push"){
                stateStack.push_back(top);
            }

            if(tokens[0] == "pop"){
                //Stack.clear();
                top = stateStack.back();
                stateStack.pop_back();
                Stack.push_back(top);
            }

            if(tokens[0] == "end"){

                break;
            }
        }

        idx++;
    }

    input.close();

    ///stage 4 input
    cout<<"***************stage 4 inp"<<endl;
    filename = "config.txt";


    input.open(filename, ios::in|ios::out|ios::app);
    idx = 0;

    while(input){
        getline(input,line);
        vector <string> tokens;
        stringstream check(line);
        string word;

        ///tokenizing input lines
        while(getline(check, word, ' '))
        {
            tokens.push_back(word);
        }

        if(idx == 0){
            screen_width = stof(tokens[0]);
            screen_height = stof(tokens[1]);
        }

        if(idx == 1){
            leftlimitX = stof(tokens[0]);
            rightlimitX = -leftlimitX;
        }

        if(idx == 2){
            bottomlimitY = stof(tokens[0]);
            toplimitY = -bottomlimitY;
        }

        if(idx == 3){
            frontZ = stof(tokens[0]);
            rearZ = stof(tokens[1]);
        }

        /*for(int i=0; i<tokens.size();i++){
            cout<<tokens[i]<<endl;
        }*/
        //cout<<line<<endl;
        idx++;
    }

    input.close();

    ///stage 4 processing
    bitmap_image image(screen_width,screen_height);
    double z_buffer[screen_height][screen_width];
    for(int i=0; i<screen_height;i++){
        for(int j=0; j<screen_width; j++){
            z_buffer[i][j] = rearZ;
        }
    }

    double dx = (rightlimitX-leftlimitX)/screen_width;
    //cout<<"dx "<<dx<<endl;
    double dy = (toplimitY-bottomlimitY)/screen_height;
    //cout<<"dy "<<dy<<endl;

    double top_Y = toplimitY - dy/2;
    double bottom_Y = bottomlimitY + dy/2;
    double left_X = leftlimitX + dx/2;
    double right_X = rightlimitX - dx/2;

    for(int i=0; i<transTrianglesS3.size(); i++){

        triangle tr = transTrianglesS3[i];
        int top_scanline, bottom_scanline;
        double max_y, min_y;
        max_y = min(getMaxY(tr),top_Y);
        min_y = max(getMinY(tr),bottom_Y);

        top_scanline = round((top_Y-max_y)/dy);
        bottom_scanline = round((top_Y-min_y)/dy);
        //cout<<i<<endl;
        cout<<top_scanline<<endl;
        cout<<bottom_scanline<<endl;
        for(int j= top_scanline+1; j<bottom_scanline;j++){
            double min_x = max(getMinX(tr),left_X);
            double max_x = min(getMaxX(tr),right_X);
            //cout<<"minX "<<min_x<<endl;
            //cout<<"maxX "<<max_x<<endl;

            double leftIntersectX, rightIntersectX,leftIntersectZ, rightIntersectZ;     ///xa,xb,za,zb
            double za, zb, zc,xa,xb,xc;
            double ys = top_Y -j*dy;            ///y val at the scanline

            if((tr.b.y-tr.a.y) != 0)        ///a intersects a\b
                za = tr.a.z + (ys - tr.a.y)*(tr.b.z-tr.a.z)/(tr.b.y-tr.a.y);
            if((tr.c.y-tr.a.y)!=0)        ///b ints a\c
                zb = tr.a.z + (ys - tr.a.y)*(tr.c.z-tr.a.z)/(tr.c.y-tr.a.y);
            if((tr.c.y-tr.b.y)!=0)        ///c ints b\c
                zc = tr.b.z + (ys - tr.b.y)*(tr.c.z-tr.b.z)/(tr.c.y-tr.b.y);

            if((tr.b.y-tr.a.y)!= 0)
                xa = tr.a.x + (ys - tr.a.y)*(tr.b.x-tr.a.x)/(tr.b.y-tr.a.y);
            if((tr.c.y-tr.a.y)!=0)
                xb = tr.a.x + (ys - tr.a.y)*(tr.c.x-tr.a.x)/(tr.c.y-tr.a.y);
            if((tr.c.y-tr.b.y)!=0)
                xc = tr.b.x + (ys - tr.b.y)*(tr.c.x-tr.b.x)/(tr.c.y-tr.b.y);

            //cout<<"point a: "<<xa<<" "<<ys<<" "<<za<<endl;
            //cout<<"point b: "<<xb<<" "<<ys<<" "<<zb<<endl;
            //cout<<"point a: "<<xc<<" "<<ys<<" "<<zc<<endl;

            int intCount = 0;
            struct point pointa, pointb, pointc;        ///intersection points
            pointa.x=xa;
            pointa.y=ys;
            pointa.z=za;

            pointb.x=xb;
            pointb.y=ys;
            pointb.z=zb;

            pointc.x=xc;
            pointc.y=ys;
            pointc.z=zc;


            if(getPointDist(tr.a,pointa) + getPointDist(tr.b,pointa) - getPointDist(tr.a,tr.b) < .001){
                intCount++;
                leftIntersectX = xa;
                leftIntersectZ = za;
            }

            if( getPointDist(tr.a,pointb)+getPointDist(tr.c,pointb) - getPointDist(tr.a,tr.c) < .001){
                intCount++;
                if(intCount==1){
                    leftIntersectX = xb;
                    leftIntersectZ = zb;
                }if(intCount==2){
                    rightIntersectX = xb;
                    rightIntersectZ = zb;
                }
            }
            if(intCount !=2 && (getPointDist(tr.c,pointc)+getPointDist(tr.b,pointc)-getPointDist(tr.c,tr.b) < .001)){
                rightIntersectX = xc;
                rightIntersectZ = zc;
                intCount++;
            }

            if(rightIntersectX<leftIntersectX){
                double tempX, tempZ;
                tempX = leftIntersectX;
                tempZ =leftIntersectZ;
                leftIntersectX = rightIntersectX;
                leftIntersectZ = rightIntersectZ;
                rightIntersectX = tempX;
                rightIntersectZ = tempZ;
            }
            //cout<<j<<endl<<leftIntersectX<<endl<<rightIntersectX<<endl;
            int leftCol, rightCol;
            leftCol = round((leftIntersectX-left_X)/dx);
            if(rightIntersectX>right_X){
                rightCol = screen_width;
            }else
                rightCol = round((rightIntersectX-left_X)/dx);
            if(leftCol<0)   leftCol = 0;



            for(int k=leftCol; k<rightCol;k++){
                //cout<<k<<endl;
                double zp, xp;
                xp = left_X+k*dx;
                zp = leftIntersectZ + (xp-leftIntersectX)*(rightIntersectZ-leftIntersectZ)/(rightIntersectX-leftIntersectX);

                if(zp < z_buffer[j][k] && zp >= frontZ){
                    z_buffer[j][k] = zp;
                    image.set_pixel(k,j,tr.color[0],tr.color[1],tr.color[2]);
                }
            }
        }
    }


    ///stage 1 out
    ofstream outfile;
    outfile.open ("stage1.txt");
    //outfile << "Writing this to a file.\n";
    for(int i=0; i<transTriangles.size(); i++){
        outfile<<strPoint(transTriangles[i].a)<<endl;
        outfile<<strPoint(transTriangles[i].b)<<endl;
        outfile<<strPoint(transTriangles[i].c)<<endl;
        outfile<<endl;
    }
    outfile.close();

    ///stage 2 out
    //ofstream outfile;
    outfile.open ("stage2.txt");
    //outfile << "Writing this to a file.\n";
    for(int i=0; i<transTrianglesS2.size(); i++){
        outfile<<strPoint(transTrianglesS2[i].a)<<endl;
        outfile<<strPoint(transTrianglesS2[i].b)<<endl;
        outfile<<strPoint(transTrianglesS2[i].c)<<endl;
        outfile<<endl;
    }
    outfile.close();

    ///stage 3 out
    //ofstream outfile;
    outfile.open ("stage3.txt");
    //outfile << "Writing this to a file.\n";
    for(int i=0; i<transTrianglesS3.size(); i++){
        outfile<<strPoint(transTrianglesS3[i].a)<<endl;
        outfile<<strPoint(transTrianglesS3[i].b)<<endl;
        outfile<<strPoint(transTrianglesS3[i].c)<<endl;
        outfile<<endl;
    }
    outfile.close();

    ///stage 4 out
    outfile.open ("z_buffer.txt");
    //outfile << "Writing this to a file.\n";
    for(int i=0; i<screen_height;i++){
        for(int j=0; j<screen_width;j++){
            if(z_buffer[i][j]!=rearZ){
                outfile<<z_buffer[i][j]<<" ";
            }else{
                outfile<<"\t";
            }
        }
        outfile<<endl;
    }
    outfile.close();
    image.save_image("test.bmp");

    return 0;
}
