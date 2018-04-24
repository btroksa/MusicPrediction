import java.io._
import scala.io.Source;

import org.apache.spark.HashPartitioner
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD

/** Genres: Rock, Rap, Latin, Jazz, Electronic, Pop, Metal, Country, Reggae, RnB, Blues, Folk, Punk, World, New Age **/

class evalGenreAnnotations extends PartialFunction [String, String] with Serializable{

  var filter = ""

  val validate: PartialFunction[Array[String], String] = {
    case lines if lines.length == 2 && lines(1).equals(this.filter) => lines(0)
  }

  override def apply(log: String): String = {
    validate(log.split("\t"))
  }

  override def isDefinedAt(x: String): Boolean = {
    validate.isDefinedAt(x.split("\t"))
  }

}

object FilterGenre {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("tester").master("local").getOrCreate()
    val textFile = spark.read.textFile("/Users/devindennis1/Documents/CS455/MusicPredictionDataSet/msd_tagtraum_cd2c.cls").rdd

    val validator = new evalGenreAnnotations
    validator.filter = args(0)
    println("Filter By Genre: " + validator.filter + "\n")

    val trackIDs = textFile.collect(validator)
    trackIDs.foreach(println)

    // create an empty map -- var map = scala.collection.mutable.Map[String, String]()

  }

}
