# backend/database_manager.py
import logging
import random
import sqlite3
import threading
import uuid
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DatabaseManager:
    """Manages interactions with the SQLite database."""

    def __init__(self, db_path: str):
        """Initializes the DatabaseManager with the database file path.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._local = threading.local()  # Create a thread-local storage

    @property
    def conn(self):
        """Gets the database connection for the current thread."""
        if not hasattr(self._local, "conn"):
            self._local.conn = self._create_connection()
        return self._local.conn

    def _create_connection(self):
        """Creates a new database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn

    def connect(self):
        """Establishes a connection to the database and creates tables if they don't exist."""
        try:
            if not hasattr(self._local, "conn"):
                self._local.conn = self._create_connection()
            self.create_tables()
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            raise

    def close(self):
        """Closes the database connection for the current thread if it exists."""
        if hasattr(self._local, "conn"):
            try:
                self._local.conn.close()
                logging.debug(
                    f"Closed DB connection for thread {threading.get_ident()}"
                )
                del self._local.conn
            except Exception as e:
                logging.error(
                    f"Error closing DB connection for thread {threading.get_ident()}: {e}"
                )
        else:
            logging.debug(
                f"No active DB connection to close for thread {threading.get_ident()}"
            )

    def create_tables(self):
        """Creates the necessary tables if they don't exist. Should only be called once."""
        conn = self._create_connection()
        cursor = conn.cursor()
        try:
            cursor.executescript(
                """
              CREATE TABLE IF NOT EXISTS samples (
                  id TEXT PRIMARY KEY,
                  path TEXT NOT NULL,
                  storage_index INTEGER,      -- Index in the consolidated .npy files
                  is_active BOOLEAN DEFAULT TRUE, -- For soft deletes if needed later
                  class_id TEXT,
                  mask_id TEXT,
                  FOREIGN KEY (class_id) REFERENCES classes(id),
                  FOREIGN KEY (mask_id)  REFERENCES masks(id)
              );

              CREATE TABLE IF NOT EXISTS clusters (
                  id TEXT PRIMARY KEY,
                  color TEXT
              );

              CREATE TABLE IF NOT EXISTS samples_clusters (
                  sample_id TEXT,
                  cluster_id TEXT,
                  PRIMARY KEY (sample_id, cluster_id),
                  FOREIGN KEY (sample_id) REFERENCES samples(id),
                  FOREIGN KEY (cluster_id) REFERENCES clusters(id)
              );

              CREATE TABLE IF NOT EXISTS classes (
                  id TEXT PRIMARY KEY,
                  name TEXT NOT NULL UNIQUE,
                  color TEXT
              );

              CREATE TABLE IF NOT EXISTS samples_classes (
                sample_id TEXT,
                class_id TEXT,
                PRIMARY KEY(sample_id, class_id),
                FOREIGN KEY (sample_id) REFERENCES samples(id),
                FOREIGN KEY (class_id) REFERENCES classes(id)
              );

              CREATE TABLE IF NOT EXISTS masks (
                  id TEXT PRIMARY KEY,
                  image_id TEXT NOT NULL,
                  masked_image_path TEXT, 
                  FOREIGN KEY (image_id) REFERENCES samples(id)
              );

              CREATE TABLE IF NOT EXISTS mask_attributes (
                  mask_id TEXT,
                  attribute_name TEXT,
                  attribute_value TEXT,
                  PRIMARY KEY (mask_id, attribute_name),
                  FOREIGN KEY (mask_id) REFERENCES masks(id)
              );

              CREATE TABLE IF NOT EXISTS plots (
                  id TEXT PRIMARY KEY,
                  name TEXT,
                  chart_type TEXT NOT NULL,       -- e.g., "scatter", "histogram"
                  parameters_json TEXT NOT NULL,  -- JSON string of chart parameters
                  created_at TEXT NOT NULL        -- ISO timestamp
              );

              CREATE TABLE IF NOT EXISTS gates (
                  id TEXT PRIMARY KEY,
                  plot_id TEXT NOT NULL,          -- Foreign key to plots table
                  name TEXT,
                  gate_type TEXT NOT NULL,        -- e.g., "rectangular", "polygon", "interval"
                  definition_json TEXT NOT NULL,  -- JSON string of gate's geometric definition
                  color TEXT,                     -- Hex color string for the gate
                  parameters_tuple_json TEXT,     -- JSON string of tuple of parameter names (e.g., ["FSC-A", "SSC-A"])
                  FOREIGN KEY (plot_id) REFERENCES plots(id) ON DELETE CASCADE -- Delete gates if plot is deleted
              );

              CREATE INDEX IF NOT EXISTS idx_samples_storage_index ON samples(storage_index);
              CREATE INDEX IF NOT EXISTS idx_samples_is_active ON samples(is_active);
              CREATE INDEX IF NOT EXISTS idx_masks_image_id ON masks(image_id);
              CREATE INDEX IF NOT EXISTS idx_samples_clusters_sample_id ON samples_clusters(sample_id);
              CREATE INDEX IF NOT EXISTS idx_samples_clusters_cluster_id ON samples_clusters(cluster_id);
              CREATE INDEX IF NOT EXISTS idx_samples_class_id on samples(class_id);
              CREATE INDEX IF NOT EXISTS idx_samples_classes_sample_id on samples_classes(sample_id);
              CREATE INDEX IF NOT EXISTS idx_samples_classes_class_id on samples_classes(class_id);
              CREATE INDEX IF NOT EXISTS idx_gates_plot_id ON gates(plot_id); -- Index for gates by plot_id
          """
            )
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error during table creation: {e}")
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    def _execute_query(self, query: str, params: tuple = (), fetchone: bool = False):
        """Helper function to execute SQL queries."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(query, params)
            self.conn.commit()
            if fetchone:
                return cursor.fetchone()
            else:
                return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Database error: {e}, Query: {query}, Params: {params}")
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    #  Sample Management
    def create_sample(
        self,
        sample_id: str,
        path: str,
        storage_index: Optional[int] = None,
        class_id: Optional[str] = None,
        mask_id: Optional[str] = None,
    ) -> None:
        """Creates a new sample record in the database."""
        # is_active defaults to TRUE in schema
        query = "INSERT INTO samples (id, path, storage_index, class_id, mask_id) VALUES (?, ?, ?, ?, ?)"
        self._execute_query(query, (sample_id, path, storage_index, class_id, mask_id))

    def create_samples(self, samples_data: List[tuple]):
        """
        Inserts multiple sample records in one transaction.
        'samples_data' should be a list of tuples, each containing:
        (id, path, storage_index, class_id, mask_id)
        """
        query = "INSERT INTO samples (id, path, storage_index, class_id, mask_id) VALUES (?, ?, ?, ?, ?)"
        cursor = self.conn.cursor()
        cursor.executemany(query, samples_data)
        self.conn.commit()
        cursor.close()

    def get_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves an active sample record by its ID."""
        query = "SELECT * FROM samples WHERE id = ? AND is_active = TRUE"
        result = self._execute_query(query, (sample_id,), fetchone=True)
        return dict(result) if result else None

    def get_all_samples(self, include_inactive=False) -> List[Dict[str, Any]]:
        """Retrieves all sample records, optionally including inactive ones."""
        if include_inactive:
            query = "SELECT * FROM samples ORDER BY storage_index ASC"
            results = self._execute_query(query)
        else:
            query = "SELECT * FROM samples WHERE is_active = TRUE ORDER BY storage_index ASC"
            results = self._execute_query(query)
        return [dict(row) for row in results] if results else []

    def update_sample_storage_index(self, sample_id: str, storage_index: int) -> None:
        """Updates the storage_index for a sample."""
        query = "UPDATE samples SET storage_index = ? WHERE id = ?"
        self._execute_query(query, (storage_index, sample_id))

    def update_sample_class_id(self, sample_id: str, class_id: Optional[str]) -> None:
        """Updates the class_id for a sample."""
        query = "UPDATE samples SET class_id = ? WHERE id = ?"
        self._execute_query(query, (class_id, sample_id))

    def update_sample_mask_id(self, sample_id: str, mask_id: Optional[str]) -> None:
        """Updates the mask_id for a sample."""
        query = "UPDATE samples SET mask_id = ? WHERE id = ?"
        self._execute_query(query, (mask_id, sample_id))

    def set_sample_active_status(self, sample_id: str, is_active: bool) -> None:
        """Sets the is_active status for a sample (for soft delete)."""
        query = "UPDATE samples SET is_active = ? WHERE id = ?"
        self._execute_query(query, (is_active, sample_id))

    def delete_sample_permanently(self, sample_id: str) -> None:
        """Deletes a sample record by its ID (hard delete)."""
        # Note: This is a hard delete. If soft delete is primary, this might be for cleanup.
        # Ensure related data (samples_clusters, samples_classes) is handled or cascaded.
        # For simplicity, assuming DataManager handles removing those links before calling this.
        query = "DELETE FROM samples WHERE id = ?"
        self._execute_query(query, (sample_id,))

    #  Feature Management
    # All feature-specific DB methods (save_features, get_image_features_path, delete_features, get_all_feature_paths)
    # are REMOVED because features are now stored in a single .npy file and indexed by samples.storage_index.
    # DataManager will handle loading/saving the .npy file and mapping features to samples.

    #  Cluster Management (largely unchanged, but sample retrieval needs to consider is_active)

    def create_cluster(self, cluster_id: str, color: str = "#FFFFFF") -> None:
        query = "INSERT INTO clusters (id, color) VALUES (?, ?)"
        self._execute_query(query, (cluster_id, color))

    def get_cluster(self, cluster_id: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM clusters WHERE id = ?"
        result = self._execute_query(query, (cluster_id,), fetchone=True)
        return dict(result) if result else None

    def get_all_clusters(self) -> List[Dict[str, Any]]:
        query = "SELECT * FROM clusters"
        results = self._execute_query(query)
        return [dict(row) for row in results]

    def delete_cluster(self, cluster_id: str) -> None:
        query_del_links = "DELETE FROM samples_clusters WHERE cluster_id = ?"
        self._execute_query(query_del_links, (cluster_id,))
        query_del_cluster = "DELETE FROM clusters WHERE id = ?"
        self._execute_query(query_del_cluster, (cluster_id,))

    def add_sample_to_cluster(self, sample_id: str, cluster_id: str) -> None:
        query = "INSERT INTO samples_clusters (sample_id, cluster_id) VALUES (?, ?)"
        try:
            self._execute_query(query, (sample_id, cluster_id))
        except sqlite3.IntegrityError:
            logging.warning(f"Sample {sample_id} is already in Cluster {cluster_id}")

    def remove_sample_from_cluster(self, sample_id: str, cluster_id: str) -> None:
        query = "DELETE FROM samples_clusters WHERE sample_id = ? AND cluster_id = ?"
        self._execute_query(query, (sample_id, cluster_id))

    def get_clusters_for_sample(self, sample_id: str) -> List[str]:
        query = "SELECT cluster_id FROM samples_clusters WHERE sample_id = ?"
        results = self._execute_query(query, (sample_id,))
        return [row["cluster_id"] for row in results] if results else []

    def get_samples_by_cluster(self, cluster_id: str) -> List[Dict[str, Any]]:
        """Retrieves all ACTIVE samples associated with a given cluster."""
        query = """
            SELECT s.*
            FROM samples s
            INNER JOIN samples_clusters sc ON s.id = sc.sample_id
            WHERE sc.cluster_id = ? AND s.is_active = TRUE
            ORDER BY s.storage_index ASC
        """
        results = self._execute_query(query, (cluster_id,))
        return [dict(row) for row in results] if results else []

    def merge_clusters(self, cluster_ids: List[str]) -> Optional[str]:
        """Merges multiple clusters into a new cluster."""
        if len(cluster_ids) < 2:
            print("Need at least two clusters to merge.")
            return None
        new_cluster_id = str(uuid.uuid4())
        new_cluster_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        self.create_cluster(new_cluster_id, new_cluster_color)
        try:
            for cluster_id in cluster_ids:
                sample_ids = [
                    sample["id"] for sample in self.get_samples_by_cluster(cluster_id)
                ]  # Already gets active samples
                for sample_id in sample_ids:
                    self.add_sample_to_cluster(sample_id, new_cluster_id)
            for cluster_id in cluster_ids:
                self.delete_cluster(cluster_id)
        except Exception as e:
            logging.error(f"Error merging clusters: {e}", exc_info=True)
            self.conn.rollback()  # Ensure rollback on error
            # Attempt to delete the newly created cluster if merge fails
            try:
                self.delete_cluster(new_cluster_id)
            except Exception as e_del:
                logging.error(
                    f"Failed to cleanup new cluster {new_cluster_id} after merge error: {e_del}"
                )
            return None
        return new_cluster_id

    #  Class Management (largely unchanged, but sample retrieval needs to consider is_active)

    def create_class(self, class_id: str, name: str, color: str = "#FFFFFF") -> None:
        query = "INSERT INTO classes (id, name, color) VALUES (?, ?, ?)"
        self._execute_query(query, (class_id, name, color))

    def get_class(self, class_id: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM classes WHERE id = ?"
        result = self._execute_query(query, (class_id,), fetchone=True)
        return dict(result) if result else None

    def get_class_by_name(self, class_name: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM classes WHERE name = ?"
        result = self._execute_query(query, (class_name,), fetchone=True)
        return dict(result) if result else None

    def get_all_classes(self) -> List[Dict[str, Any]]:
        query = "SELECT * FROM classes"
        results = self._execute_query(query)
        return [dict(row) for row in results] if results else []

    def add_images_to_class(self, image_ids: List[str], class_id: str) -> None:
        """Adds images to a specified class and updates samples.class_id."""
        insert_junction_query = (
            "INSERT OR IGNORE INTO samples_classes (sample_id, class_id) VALUES (?, ?)"
        )
        update_sample_query = "UPDATE samples SET class_id = ? WHERE id = ?"

        operations = []
        for image_id in image_ids:
            operations.append((insert_junction_query, (image_id, class_id)))
            operations.append((update_sample_query, (class_id, image_id)))

        cursor = self.conn.cursor()
        try:
            for query, params in operations:
                cursor.execute(query, params)
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(
                f"Error adding images to class {class_id}: {e}", exc_info=True
            )
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def remove_images_from_class(self, image_ids: List[str], class_id: str) -> None:
        """Removes images from a specified class and sets samples.class_id to NULL."""
        delete_junction_query = (
            "DELETE FROM samples_classes WHERE sample_id = ? AND class_id = ?"
        )
        # Prepare a string of placeholders for the IN clause
        placeholders = ",".join(["?"] * len(image_ids))
        update_sample_query = (
            f"UPDATE samples SET class_id = NULL WHERE id IN ({placeholders})"
        )

        operations = []
        for image_id in image_ids:
            operations.append((delete_junction_query, (image_id, class_id)))

        # The update_sample_query uses the list of image_ids directly
        operations.append((update_sample_query, tuple(image_ids)))

        cursor = self.conn.cursor()
        try:
            for query, params in operations:
                cursor.execute(query, params)
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(
                f"Error removing images from class {class_id}: {e}", exc_info=True
            )
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def delete_class(self, class_id: str) -> None:
        """Deletes a class and sets samples.class_id to NULL for associated images."""
        # First, delete all links in the junction table
        query_junction = "DELETE FROM samples_classes WHERE class_id = ?"
        self._execute_query(query_junction, (class_id,))
        # Then, delete the class itself
        query_class = "DELETE FROM classes WHERE id = ?"
        self._execute_query(query_class, (class_id,))
        # Set class_id in samples to null
        query_samples = "UPDATE samples SET class_id = NULL WHERE class_id = ?"
        self._execute_query(query_samples, (class_id,))

    def get_images_by_class(self, class_id: str) -> List[Dict[str, Any]]:
        """Retrieves all ACTIVE image data associated with a given class."""
        query = """
          SELECT s.*
          FROM samples s
          INNER JOIN samples_classes sc ON s.id = sc.sample_id
          WHERE sc.class_id = ? AND s.is_active = TRUE
          ORDER BY s.storage_index ASC
      """
        results = self._execute_query(query, (class_id,))
        return [dict(row) for row in results] if results else []

    def rename_class(self, class_id: str, new_name: str) -> None:
        """Renames a class."""
        query = "UPDATE classes SET name = ? WHERE id = ?"
        self._execute_query(query, (new_name, class_id))

    #  Mask Management

    def create_mask(
        self, mask_id: str, image_id: str, masked_image_path: Optional[str] = None
    ) -> None:
        """Creates a new mask record. Path to .npy is removed."""
        query = "INSERT INTO masks (id, image_id, masked_image_path) VALUES (?, ?, ?)"
        self._execute_query(query, (mask_id, image_id, masked_image_path))
        # Also update the sample's mask_id
        self.update_sample_mask_id(image_id, mask_id)

    def get_mask(self, mask_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a mask record by its ID."""
        query = "SELECT * FROM masks WHERE id = ?"
        result = self._execute_query(query, (mask_id,), fetchone=True)
        return dict(result) if result else None

    def delete_mask(self, mask_id: str) -> None:
        """Deletes a mask record, its attributes, and nullifies sample.mask_id."""
        # Find the image_id associated with this mask to update the sample
        mask_info = self.get_mask(mask_id)

        # First delete related entries in mask_attributes
        query_attrs = "DELETE FROM mask_attributes WHERE mask_id = ?"
        self._execute_query(query_attrs, (mask_id,))
        # Then delete the mask record
        query_mask = "DELETE FROM masks WHERE id = ?"
        self._execute_query(query_mask, (mask_id,))

        if mask_info:
            image_id_of_mask = mask_info.get("image_id")
            if image_id_of_mask:
                # Set mask_id in the corresponding sample to null
                query_sample_update = (
                    "UPDATE samples SET mask_id = NULL WHERE id = ? AND mask_id = ?"
                )
                self._execute_query(query_sample_update, (image_id_of_mask, mask_id))

    def save_mask_attributes(self, mask_id: str, attributes: Dict[str, Any]) -> None:
        """Saves mask attributes."""
        # Delete old attributes first to handle updates correctly
        del_query = "DELETE FROM mask_attributes WHERE mask_id = ?"
        self._execute_query(del_query, (mask_id,))

        insert_query = "INSERT INTO mask_attributes (mask_id, attribute_name, attribute_value) VALUES (?, ?, ?)"
        operations = []
        for name, value in attributes.items():
            value_str = (
                str(value) if not isinstance(value, (int, float, str)) else value
            )
            operations.append((insert_query, (mask_id, name, value_str)))

        cursor = self.conn.cursor()
        try:
            for query, params in operations:
                cursor.execute(query, params)
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(
                f"Error saving mask attributes for {mask_id}: {e}", exc_info=True
            )
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def get_mask_attributes(self, mask_id: str) -> Dict[str, Any]:
        """Retrieves all attributes for a given mask."""
        query = "SELECT attribute_name, attribute_value FROM mask_attributes WHERE mask_id = ?"
        results = self._execute_query(query, (mask_id,))
        attributes = {}
        for row in results:
            try:
                # Attempt to convert to float, if not, keep as string
                attributes[row["attribute_name"]] = float(row["attribute_value"])
            except (ValueError, TypeError):
                attributes[row["attribute_name"]] = row["attribute_value"]
        return attributes

    def get_highest_storage_index(self) -> int:
        """Retrieves the highest storage_index from the samples table."""
        query = "SELECT MAX(storage_index) FROM samples WHERE is_active = TRUE"
        result = self._execute_query(query, fetchone=True)
        if result and result[0] is not None:
            return int(result[0])
        return -1  # Indicates no active samples with a storage_index or table is empty

    def update_samples_storage_indices(self, id_index_map: Dict[str, int]) -> None:
        """Batch updates storage_index for multiple samples."""
        query = "UPDATE samples SET storage_index = ? WHERE id = ?"
        operations = [(index, sample_id) for sample_id, index in id_index_map.items()]

        cursor = self.conn.cursor()
        try:
            cursor.executemany(query, operations)
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error batch updating storage indices: {e}", exc_info=True)
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    def clear_all_data_for_session(self):
        """
        Deletes all data from all tables. Use with extreme caution.
        """
        tables = [
            "gates",
            "plots",  # Delete from child tables first due to FKs (or rely on cascade)
            "mask_attributes",
            "masks",
            "samples_classes",
            "classes",
            "samples_clusters",
            "clusters",
            "samples",
        ]
        cursor = self.conn.cursor()
        try:
            for table in tables:
                cursor.execute(f"DELETE FROM {table}")
            self.conn.commit()
            logging.info("All data cleared from session database.")
        except sqlite3.Error as e:
            logging.error(f"Error clearing all data: {e}", exc_info=True)
            self.conn.rollback()
            raise
        finally:
            cursor.close()

    # Gates

    def create_gate(
        self,
        gate_id: str,
        plot_id: str,
        name: str,
        gate_type: str,
        definition_json: str,
        color: str,
        parameters_tuple_json: str,
    ) -> None:
        query = "INSERT INTO gates (id, plot_id, name, gate_type, definition_json, color, parameters_tuple_json) VALUES (?, ?, ?, ?, ?, ?, ?)"
        self._execute_query(
            query,
            (
                gate_id,
                plot_id,
                name,
                gate_type,
                definition_json,
                color,
                parameters_tuple_json,
            ),
        )

    def get_gate(self, gate_id: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM gates WHERE id = ?"
        result = self._execute_query(query, (gate_id,), fetchone=True)
        return dict(result) if result else None

    def get_all_gates(self) -> List[Dict[str, Any]]:
        query = "SELECT * FROM gates ORDER BY plot_id ASC, name ASC"
        results = self._execute_query(query)
        return [dict(row) for row in results] if results else []

    def get_all_plots(self) -> List[Dict[str, Any]]:
        query = "SELECT * FROM plots"
        results = self._execute_query(query)
        return [dict(row) for row in results] if results else []

    def get_gates_for_plot(self, plot_id: str) -> List[Dict[str, Any]]:
        query = "SELECT * FROM gates WHERE plot_id = ?"
        results = self._execute_query(query, (plot_id,))
        return [dict(row) for row in results] if results else []

    def update_gate(
        self,
        gate_id: str,
        name: Optional[str] = None,
        definition_json: Optional[str] = None,
        color: Optional[str] = None,
        parameters_tuple_json: Optional[str] = None,
    ) -> None:
        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if definition_json is not None:
            updates.append("definition_json = ?")
            params.append(definition_json)
        if color is not None:
            updates.append("color = ?")
            params.append(color)
        if parameters_tuple_json is not None:
            updates.append("parameters_tuple_json = ?")
            params.append(parameters_tuple_json)

        if not updates:
            return

        query = f"UPDATE gates SET {', '.join(updates)} WHERE id = ?"
        params.append(gate_id)
        self._execute_query(query, tuple(params))

    def delete_gate(self, gate_id: str) -> None:
        query = "DELETE FROM gates WHERE id = ?"
        self._execute_query(query, (gate_id,))

    def delete_gates_for_plot(self, plot_id: str) -> None:
        # This might be redundant if ON DELETE CASCADE is reliable on the plot_id foreign key.
        # But explicit deletion can be safer or useful if cascade is disabled for some reason.
        query = "DELETE FROM gates WHERE plot_id = ?"
        self._execute_query(query, (plot_id,))

    #  Plot Management
    def create_plot(
        self,
        plot_id: str,
        name: str,
        chart_type: str,
        parameters_json: str,
        created_at: str,
    ) -> None:
        query = "INSERT INTO plots (id, name, chart_type, parameters_json, created_at) VALUES (?, ?, ?, ?, ?)"
        self._execute_query(
            query, (plot_id, name, chart_type, parameters_json, created_at)
        )

    def get_plot(self, plot_id: str) -> Optional[Dict[str, Any]]:
        query = "SELECT * FROM plots WHERE id = ?"
        result = self._execute_query(query, (plot_id,), fetchone=True)
        return dict(result) if result else None

    def get_all_plots(self) -> List[Dict[str, Any]]:
        query = "SELECT * FROM plots ORDER BY created_at ASC"
        results = self._execute_query(query)
        return [dict(row) for row in results] if results else []

    def update_plot(
        self,
        plot_id: str,
        name: Optional[str] = None,
        parameters_json: Optional[str] = None,
    ) -> None:
        updates = []
        params = []
        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if parameters_json is not None:
            updates.append("parameters_json = ?")
            params.append(parameters_json)

        if not updates:
            return

        query = f"UPDATE plots SET {', '.join(updates)} WHERE id = ?"
        params.append(plot_id)
        self._execute_query(query, tuple(params))

    def delete_plot(self, plot_id: str) -> None:
        # Gates associated with this plot will be deleted due to ON DELETE CASCADE
        query = "DELETE FROM plots WHERE id = ?"
        self._execute_query(query, (plot_id,))
