import java.io._

import org.apache.spark.HashPartitioner
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD

object FilterGenre {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder.appName("tester").master("local").getOrCreate()
    val textFile = spark.read.textFile("MillionSongSubset/data/A/A/A/TRAAAAW128F429D538.h5").rdd
    textFile.foreach(println)


  }


}
