#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
using namespace std;


double cosine_similarity(int a1[50], int a2[50]) {
    int numer = 0;
    int denom1 = 0;
    int denom2 = 0;
    for (int x=0; x < 50; x++) {
        numer += (a1[x] * a2[x]);
        denom1 += pow(a1[x],2.0);
        denom2 += pow(a2[x],2.0);
    };
    return numer / (pow(denom1,0.5) * pow(denom2,0.5));
};

// Returns True if time1 is earlier than time2
bool time_check(string time1, string time2) {

    int i;
    string y1;
    string y2;
    string m1;
    string m2;
    string d1;
    string d2;

    for (i=0;i<time1.length();i++) {
      if (i == 2 || i == 5) continue;
      if (i < 2) {
        m1 += time1[i];
      } else if (i < 5) {
        d1 += time1[i];
      } else {
        y1 += time1[i];
      };
    };

    for (i=0;i<time2.length();i++) {
      if (i == 2 || i == 5) continue;
      if (i < 2) {
        m2 += time2[i];
      } else if (i < 5) {
        d2 += time2[i];
      } else {
        y2 += time2[i];
      };
    };

    int year1 = 0;
    int year2 = 0;
    int month1 = 0;
    int month2 = 0;
    int day1 = 0;
    int day2 = 0;

    stringstream year1_i(y1);
    stringstream year2_i(y2);
    stringstream month1_i(m1);
    stringstream month2_i(m2);
    stringstream day1_i(d1);
    stringstream day2_i(d2);

    year1_i >> year1;
    year2_i >> year2;
    month1_i >> month1;
    month2_i >> month2;
    day1_i >> day1;
    day2_i >> day2;

    if (year2 < year1) return false;
    if (year2 > year1) return true;
    if (month2 < month1) return false;
    if (month2 > month1) return true;
    if (day2 < day1) return false;
    if (day2 > day1) return true;

    return false;
};





int compare_patterns(string start_date, float threshold, int curr_pattern[50]) {
    string STRING;
    ifstream infile;
    int i;
    int j=0;
    int matches = 0;

    infile.open("all_patterns.csv");
        while(!infile.eof())
        {
            // cout << j << endl;
            j += 1;

            getline(infile,STRING);
            // cout << STRING << endl;

            string date;
            for (i=0;i<STRING.length();i++) {
              // cout << STRING[i] << endl;
              if (STRING[i] == ',') break;
              date += STRING[i];
            };

            string time;
            for (i=i+1;i<STRING.length();i++) {
                if (STRING[i] == ',') break;
                time += STRING[i];
            };
            int time_i;
            stringstream time_x(time);
            time_x >> time_i;

            if (time_check(start_date,date)) break;

            string num;
            int pattern[50] = {0};
            int p = 0;
            for (i=i+3;i<STRING.length();i++) {
                if (STRING[i] == ' ') continue;
                if (STRING[i] == ']' || STRING[i] == '"') break;
                if (STRING[i] == ',') {
                  float num_i;
                  stringstream num_x(num);
                  num_x >> num_i;
                  // cout << num_i << endl;
                  pattern[p] = num_i;
                  p += 1;
                  num.clear();

                } else {
                  num += STRING[i];
                }
            };

            num.clear();
            int outcome[20] = {0};
            p = 0;
            for (i=i+5;i<STRING.length();i++) {
                if (STRING[i] == ' ') continue;
                if (STRING[i] == ']' || STRING[i] == '"') break;
                if (STRING[i] == ',') {
                  float num_i;
                  stringstream num_x(num);
                  num_x >> num_i;
                  // cout << num_i << endl;
                  outcome[p] = num_i;
                  p += 1;
                  num.clear();

                } else {
                  num += STRING[i];
                }
            };

            float sim = cosine_similarity(pattern, curr_pattern);
            if (sim > threshold) {
              matches += 1;
            };
            // cout << date << endl << time_i << endl;

            // break;
        };
    infile.close();
    return matches;
};






int main()
{
    string start_d("02/10/18");
    string STRING;
  	ifstream infile;
    int i;
    int j=0;

  	infile.open("all_patterns.csv");
        while(!infile.eof())
        {
            // cout << j << endl;
            j += 1;

            getline(infile,STRING);
            // cout << STRING << endl;

            string date;
            for (i=0;i<STRING.length();i++) {
              // cout << STRING[i] << endl;
              if (STRING[i] == ',') break;
              date += STRING[i];
            };

            if (time_check(date,start_d)) continue;

            string time;
            for (i=i+1;i<STRING.length();i++) {
                if (STRING[i] == ',') break;
                time += STRING[i];
            };
            int time_i;
            stringstream time_x(time);
            time_x >> time_i;


            string num;
            int pattern[50] = {0};
            int p = 0;
            for (i=i+3;i<STRING.length();i++) {
                if (STRING[i] == ' ') continue;
                if (STRING[i] == ']' || STRING[i] == '"') break;
                if (STRING[i] == ',') {
                  float num_i;
                  stringstream num_x(num);
                  num_x >> num_i;
                  // cout << num_i << endl;
                  pattern[p] = num_i;
                  p += 1;
                  num.clear();

                } else {
                  num += STRING[i];
                }
            };

            num.clear();
            int outcome[20] = {0};
            p = 0;
            for (i=i+5;i<STRING.length();i++) {
                if (STRING[i] == ' ') continue;
                if (STRING[i] == ']' || STRING[i] == '"') break;
                if (STRING[i] == ',') {
                  float num_i;
                  stringstream num_x(num);
                  num_x >> num_i;
                  // cout << num_i << endl;
                  outcome[p] = num_i;
                  p += 1;
                  num.clear();

                } else {
                  num += STRING[i];
                }
            };

            int matches = compare_patterns(date,.9,pattern);
            cout << date << endl;
            cout << time << endl << endl;
            cout << matches << endl << endl;
            // cout << date << endl << time_i << endl;
            // time_check(date,date);
            // break;
        };
  	infile.close();
};













//
