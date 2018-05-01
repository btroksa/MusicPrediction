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

  override def apply(line: String): String = {
    validate(line.split("\t"))
  }

  override def isDefinedAt(line: String): Boolean = {
    validate.isDefinedAt(line.split("\t"))
  }

}

object FilterGenre {

  var genreSet = scala.collection.mutable.HashSet[String]()
  var midiResultSet = scala.collection.mutable.HashSet[String]()

  def traverse(dir: File): Unit =
    dir.listFiles.foreach({
            f =>  val item = f.toString.substring(f.toString.lastIndexOf("/") + 1, f.toString.length)
                  if (f.isDirectory && item.length == 18 && this.genreSet.contains(item)) traverse(f)
                  else if (f.isDirectory && item.length == 18) doNothing(f)
                  else if(item.equals(".DS_Store")) doNothing(f)
                  else if (f.isDirectory && item.length() == 1) traverse(f)
                  else process(f)
        })

  def doNothing(File: File) = {}

  def process(File: File) ={
    if(!midiResultSet.contains(File.toString)) midiResultSet.add(File.toString)
  }

  /** sbt shell: run <Genre> **/
  def main(args: Array[String]): Unit = {

    val path = args(1) // TODO: UPDATE FILE PATH
    val validator = new evalGenreAnnotations
    validator.filter = args(0)

    println("\nFilter By Genre: " + validator.filter + "\n")

    val spark = SparkSession.builder.appName("tester").master("local").getOrCreate()
    val trackIDs = spark.read.textFile(path).rdd.collect(validator)
    trackIDs.foreach(genreSet.add(_))

    val midis = args(2)          // TODO: UPDATE FILE PATH
    traverse(new File(midis))

    val file = new File(args(3))  // TODO: UPDATE FILE PATH
    val bw = new BufferedWriter(new FileWriter(file))
    for(midi <- midiResultSet){
      val line:String = midi.concat("\n")
      bw.write(line)
    }
    bw.close

  }

}
